"""
Visualize GROUND-TRUTH grasps + per-point graspness heatmap from a processed h5, using viser.

Usage:
uv run scripts/stages/data_process/visualize_processed_data.py --experiment_name g1_inspire_wbc_dex1b_v0_5_small_spconv
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
from dataclasses import dataclass, replace
from typing import Literal

import matplotlib
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
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.viser_utils import visualize_scene_meshes

logger = logging.getLogger("graspkit")


def main(
    cfg: ExperimentConfig,
    split: Literal["train", "test"],
    processed_data_path: str | None,
):
    device = torch.device("cpu")
    h5_path = processed_data_path or cfg.processed_data_path
    dataset_config = replace(cfg.dataset, return_actions=True, return_graspness=True)
    dataset = GraspDataset(
        dataset_config,
        h5_path=ensure_local_filepath(h5_path),
        scene_description_path=cfg.scene_description_path,
        split=split,
    )
    num_scenes = len(dataset)
    batch = next(iter(DataLoader(dataset, batch_size=num_scenes, shuffle=False, num_workers=0)))

    pcd_all = batch["observations"]["pointcloud"].numpy()  # (S, N, 3)
    graspness_all = batch["observations"]["graspness"].numpy()  # (S, N)
    actions_all = batch["actions"]  # (S, num_grasps, num_phases, dof)
    scene_ids = list(batch["metas"]["scene_id"])
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    scene_descs = [scene_descriptions[scene_id] for scene_id in scene_ids]
    num_grasps, num_phases = actions_all.shape[1], actions_all.shape[2]

    robot_helper = GraspRobot.from_config(cfg.robot_config, load_meshes=True, device=device)
    urdf_load_kwargs = {"build_scene_graph": True, "load_meshes": True}
    if robot_helper.config.mesh_dir is not None:
        urdf_load_kwargs["mesh_dir"] = robot_helper.config.mesh_dir
    urdf = yourdfpy.URDF.load(robot_helper.config.urdf_or_mjcf_path, **urdf_load_kwargs)

    cmap = matplotlib.colormaps["jet"]
    logger.info(f"Visualizing GT grasps + graspness over {num_scenes} {split} scenes from {h5_path}")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, shadow_opacity=0.0)
    server.scene.configure_default_lights(cast_shadow=False)
    robot_base_pos = np.array(robot_helper.config.init_position)
    robot_base_wxyz = np.array(robot_helper.config.init_quaternion)
    robot_frame_handle = server.scene.add_frame(
        "/robot_base", position=robot_base_pos, wxyz=robot_base_wxyz, axes_length=0.05, axes_radius=0.002
    )
    robot_handle = ViserUrdf(server, urdf, root_node_name="/robot_base/robot", mesh_color_override=ROBOT_COLOR)
    palm_idx = robot_helper.link_names.index(robot_helper.config.palm_link_name)
    (thumb_tip_link_name,) = set(robot_helper.config.thumb_link_names) & set(robot_helper.config.fingertip_link_names)
    (middle_tip_link_name,) = set(robot_helper.config.middle_link_names) & set(robot_helper.config.fingertip_link_names)
    thumb_idx = robot_helper.link_names.index(thumb_tip_link_name)
    middle_idx = robot_helper.link_names.index(middle_tip_link_name)
    palm_center_local = torch.as_tensor(
        robot_helper.get_link_trimesh_meshes(mode="collision")[robot_helper.config.palm_link_name].center_mass,
        dtype=torch.float,
    )

    scene_slider = server.gui.add_slider(label="Scene", min=0, max=max(1, num_scenes - 1), step=1, initial_value=0)
    grasp_slider = server.gui.add_slider(label="Grasp", min=0, max=max(1, num_grasps - 1), step=1, initial_value=0)
    phase_slider = server.gui.add_slider(label="Phase", min=0, max=max(1, num_phases - 1), step=1, initial_value=1)

    def update_viz():
        scene_idx = int(scene_slider.value)
        grasp_idx = int(grasp_slider.value)
        phase_idx = int(phase_slider.value)

        # graspness heatmap point cloud (normalize per scene to [0,1] for color)
        g = graspness_all[scene_idx]
        g_norm = g / (g.max() + 1e-6)
        colors = (cmap(g_norm)[:, :3] * 255).astype(np.uint8)
        server.scene.add_point_cloud("/graspness", points=pcd_all[scene_idx], colors=colors, point_size=0.004)
        visualize_scene_meshes(server, scene_descs[scene_idx])

        # contact / graspness point for this grasp (best points are the first num_grasps points)
        server.scene.add_point_cloud(
            "/contact",
            points=pcd_all[scene_idx, grasp_idx][None],
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            point_size=0.012,
        )

        qpos = actions_all[scene_idx, grasp_idx, phase_idx]
        if robot_helper.config.base_type == "floating":
            robot_frame_handle.position = robot_base_pos + qpos[:3].numpy()
            robot_frame_handle.wxyz = matrix_to_quaternion(euler_angles_to_matrix(qpos[3:6].numpy(), "rxyz"))
            robot_handle.update_cfg(qpos[6:].numpy())
        else:
            robot_handle.update_cfg(qpos.numpy())
            link_poses = robot_helper.forward_kinematics_torch(qpos[None])[0]  # (num_links, 4, 4)
            # contact (graspness cone) frame: x-axis = thumb-middle center -> palm center (the cone axis)
            palm_pose = link_poses[palm_idx]
            palm_center = palm_pose[:3, :3] @ palm_center_local + palm_pose[:3, 3]
            thmf = (link_poses[thumb_idx][:3, 3] + link_poses[middle_idx][:3, 3]) / 2
            x = torch.nn.functional.normalize(thmf - palm_center, dim=-1, eps=1e-8)
            z = torch.tensor([0.0, 0.0, 1.0]) - (torch.tensor([0.0, 0.0, 1.0]) @ x) * x
            z = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
            y = torch.cross(z, x, dim=-1)
            cone_R = torch.stack([x, y, z], dim=-1).numpy()
            server.scene.add_frame(
                "/robot_base/contact_axis",
                position=palm_center.numpy(),
                wxyz=matrix_to_quaternion(cone_R),
                axes_length=0.1,
                axes_radius=0.004,
            )

    for slider in (scene_slider, grasp_slider, phase_slider):
        slider.on_update(lambda _event: update_viz())

    update_viz()
    input("Press Enter to exit...")


if __name__ == "__main__":

    @dataclass
    class Args:
        experiment_name: str
        processed_data_path: str | None = None
        split: Literal["train", "test"] = "test"

    args = tyro.cli(Args, use_underscores=True)
    main(build_experiment_config(args.experiment_name), args.split, args.processed_data_path)
