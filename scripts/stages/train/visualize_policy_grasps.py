"""
Visualize grasps sampled from a trained open-loop policy using viser.

Usage:
uv run scripts/stages/train/visualize_policy_grasps.py --experiment_name tabletop_demo
uv run scripts/stages/train/visualize_policy_grasps.py --experiment_name tabletop_demo --num_grasps_per_scene 16
uv run scripts/stages/train/visualize_policy_grasps.py \
    --experiment_name ability_right_dex1b_v0_4_mini_tabletop \
    --checkpoint_path s3://<bucket>/logs/train/ability_dex1b_v0_4_tabletop_mini/checkpoints/best_eval_success_rate.pt \
    --processed_data_path data/sim_data/ability_grasps/dex1b_v0_4_tabletop_mini.h5
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from robokit.xform.numpy import euler_angles_to_matrix, matrix_to_quaternion
import tyro
import viser
import yourdfpy
from torch.utils.data import DataLoader
from viser.extras import ViserUrdf

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.constants import ROBOT_COLOR
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.viser_utils import visualize_scene_meshes

logger = logging.getLogger("graspkit")


def main(
    cfg: ExperimentConfig,
    checkpoint_path: str,
    split: Literal["train", "test"],
    num_grasps_per_scene: int,
    processed_data_path: str | None,
):
    assert cfg.agent is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_path = processed_data_path or cfg.processed_data_path
    dataset_config = replace(cfg.dataset, return_actions=False, return_graspness=False)
    dataset = GraspDataset(
        dataset_config,
        h5_path=ensure_local_filepath(h5_path),
        scene_description_path=cfg.scene_description_path,
        split=split,
    )
    num_scenes = len(dataset)
    dataloader = DataLoader(dataset, batch_size=num_scenes, shuffle=False, drop_last=False, num_workers=0)
    batch = next(iter(dataloader))

    robot_config = cfg.robot_config
    agent = build_grasp_policy(cfg.agent, robot_config, device)
    checkpoint = torch.load(ensure_local_filepath(checkpoint_path), map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    obs_all = {key: value.to(device) for key, value in batch["observations"].items()}
    scene_ids = list(batch["metas"]["scene_id"])
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    scene_descs = [scene_descriptions[scene_id] for scene_id in scene_ids]

    robot_helper = GraspRobot.from_config(robot_config, device=device)
    # The policy predicts chain-space qpos; the viz URDF wants full-robot qpos. Scatter by joint name.
    full_names = GraspRobot.from_config(robot_config, prune=False).actuated_joint_names
    chain_indices = [full_names.index(n) for n in robot_helper.actuated_joint_names]
    full_qpos = np.asarray(robot_config.init_qpos, dtype=np.float64)
    urdf_load_kwargs = {
        "build_scene_graph": True,
        "build_collision_scene_graph": True,
        "load_meshes": True,
        "load_collision_meshes": True,
    }
    if robot_helper.config.mesh_dir is not None:
        urdf_load_kwargs["mesh_dir"] = robot_helper.config.mesh_dir
    urdf = yourdfpy.URDF.load(robot_helper.config.urdf_or_mjcf_path, **urdf_load_kwargs)

    logger.info(
        f"Visualizing {num_grasps_per_scene} policy grasps per scene over {num_scenes} {split} scenes from {checkpoint_path}"
    )
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, shadow_opacity=0.0)
    server.scene.configure_default_lights(cast_shadow=False)
    robot_base_pos = np.array(robot_helper.config.init_position)
    robot_base_wxyz = np.array(robot_helper.config.init_quaternion)
    robot_frame_handle = server.scene.add_frame(
        "/robot_base",
        position=robot_base_pos,
        wxyz=robot_base_wxyz,
        axes_length=0.05,
        axes_radius=0.002,
    )
    robot_handle = ViserUrdf(
        server,
        urdf,
        root_node_name="/robot_base/robot",
        mesh_color_override=ROBOT_COLOR,
        collision_mesh_color_override=ROBOT_COLOR,
    )

    scene_slider = server.gui.add_slider(label="Scene", min=0, max=max(1, num_scenes - 1), step=1, initial_value=0)
    grasp_slider = server.gui.add_slider(
        label="Grasp", min=0, max=max(1, num_grasps_per_scene - 1), step=1, initial_value=0
    )
    phase_slider = server.gui.add_slider(label="Phase", min=0, max=1, step=1, initial_value=1)
    resample_button = server.gui.add_button("Resample policy")

    state: dict = {"trajectories": None, "scene_idx": -1}

    def eval_scene(scene_idx: int):
        scene_obs = {key: value[scene_idx : scene_idx + 1] for key, value in obs_all.items()}
        with torch.no_grad():
            actions = agent.get_action(scene_obs, num_samples_per_scene=num_grasps_per_scene)
        trajectories = actions[0]  # (grasp, phase, dof)
        state["trajectories"] = trajectories
        state["scene_idx"] = scene_idx
        num_phases = trajectories.shape[1]
        phase_slider.max = max(1, num_phases - 1)
        phase_slider.value = min(int(phase_slider.value), num_phases - 1)
        visualize_scene_meshes(server, scene_descs[scene_idx])

    def update_viz():
        scene_idx = int(scene_slider.value)
        if scene_idx != state["scene_idx"]:
            eval_scene(scene_idx)
        grasp_idx = int(grasp_slider.value)
        phase_idx = int(phase_slider.value)
        qpos = state["trajectories"][grasp_idx, phase_idx]

        if robot_helper.config.base_type == "floating":
            base_pos = qpos[:3].cpu().numpy()
            base_rot_euler = qpos[3:6].cpu().numpy()
            wxyz = matrix_to_quaternion(euler_angles_to_matrix(base_rot_euler, "rxyz"))
            hand_joints = qpos[6:].cpu().numpy()
            robot_frame_handle.position = robot_base_pos + base_pos
            robot_frame_handle.wxyz = wxyz
            robot_handle.update_cfg(hand_joints)
        else:
            full_qpos[chain_indices] = qpos.cpu().numpy()
            robot_handle.update_cfg(full_qpos)

    @scene_slider.on_update
    def _(_event):
        update_viz()

    @grasp_slider.on_update
    def _(_event):
        update_viz()

    @phase_slider.on_update
    def _(_event):
        update_viz()

    @resample_button.on_click
    def _(_event):
        eval_scene(int(scene_slider.value))
        update_viz()

    update_viz()
    input("Press Enter to exit...")


if __name__ == "__main__":

    @dataclass
    class Args:
        experiment_name: str
        checkpoint_path: str | None = None
        processed_data_path: str | None = None
        split: Literal["train", "test"] = "test"
        num_grasps_per_scene: int = 8
        use_mean: bool | None = None
        """Override CVAE sampling: True = decode the mean (z=0), False = sample z~N(0,I)."""

    args = tyro.cli(Args, use_underscores=True)
    exp_cfg = build_experiment_config(args.experiment_name)
    if args.use_mean is not None:
        exp_cfg.agent.use_mean = args.use_mean
    ckpt_path = args.checkpoint_path
    if ckpt_path is None:
        ckpt_path = str(Path("logs/train") / exp_cfg.name / "checkpoints/best_eval_success_rate.pt")
    main(exp_cfg, ckpt_path, args.split, args.num_grasps_per_scene, args.processed_data_path)
