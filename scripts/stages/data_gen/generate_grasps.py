"""
Generate grasps.

Usage:
python scripts/stages/data_gen/generate_grasps.py shadow_dex1b_v1
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import io
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import fpsample
import fsspec
import h5py
import numpy as np
import torch
import tyro
from rich.progress import track
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from torch.types import Device

from graspkit.configs.experiment_config import EnvConfig, GraspOptimConfig, GraspTrajBuilderConfig, MotionPlannerConfig
from graspkit.configs.grasp_policy import GRASP_POLICY_CONFIGS
from graspkit.envs import build_env
from graspkit.configs.env_config import MsEnvConfig
from graspkit.constants import DATA_ROOT, PROJECT_ROOT, S3_ROOT
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot import build_robot_config
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_traj_build import GraspTrajBuilder
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.seed_utils import seed_everything

torch.set_float32_matmul_precision("high")
logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)


# ---------------------------------------------------------------------------
# GraspGenerator (inlined from former graspkit.stages.grasp_generator)
# ---------------------------------------------------------------------------


@dataclass
class GraspGeneratorConfig:
    robot_config_name: str = "xarm7_ability_right"
    """Robot config name"""
    ms_robot_uid: str = "xarm7_ability_right"
    """ManiSkill agent uid for the sim evaluator (floating hands: the *_floating_base agent)."""
    agent_config_name: str = "pointnet_basic"
    """Agent config name"""
    agent_checkpoint_path: Optional[str] = None
    """Agent checkpoint path"""
    use_additional_motion_planner: bool = False
    """Whether to use an additional motion planner for real robot"""


GRASP_GENERATOR_CONFIGS: Dict[str, GraspGeneratorConfig] = {
    # ------------------------------------- ability_right ----------------------------------------------
    "ability_spconv_grasp_generator": GraspGeneratorConfig(
        robot_config_name="ability_right",
        ms_robot_uid="ability_right_floating_base",
        agent_config_name="spconv_layernorm_local_kin_loss_6drot",
        agent_checkpoint_path=str(
            S3_ROOT
            / "logs/train/ability/dex1b_v1_gen_data_cvae_spconv_local_mlp_6drot_0.1_kl/checkpoints/best_eval_success_rate.pt"
        ),
        use_additional_motion_planner=True,
    ),
    # ------------------------------------- inspire_right ----------------------------------------------
    "inspire_spconv_grasp_generator": GraspGeneratorConfig(
        robot_config_name="inspire_right",
        ms_robot_uid="inspire_right_floating_base",
        agent_config_name="spconv_local_kin_loss_6drot",
        agent_checkpoint_path=str(
            S3_ROOT
            / "logs/train/inspire/ycb_objaverse_7k_scenes_cvae_spconv_local_mlp_6drot_0.1_kl/checkpoints/best_eval_success_rate.pt"
        ),
        use_additional_motion_planner=True,
    ),
    # ------------------------------------- shadow_right ----------------------------------------------
    "shadow_spconv_grasp_generator": GraspGeneratorConfig(
        robot_config_name="shadow_right",
        ms_robot_uid="shadow_right_mj_floating_base",
        agent_config_name="spconv_layernorm_local_kin_loss_6drot",
        agent_checkpoint_path=str(
            S3_ROOT
            / "logs/train/shadow/ycb_objaverse_7k_scenes_cvae_spconv_local_mlp_6drot_0.1_kl/checkpoints/best_eval_success_rate.pt"
        ),
        use_additional_motion_planner=True,
    ),
    # ------------------------------------- xarm ability_right ------------------------------------------
    "xarm_ability_spconv_grasp_generator": GraspGeneratorConfig(
        robot_config_name="xarm7_ability_right",
        agent_config_name="spconv_layernorm_local_kin_loss",
        agent_checkpoint_path=str(
            S3_ROOT
            / "logs/train/xarm_ability/ycb_objaverse_7k_scenes_cvae_spconv_layernorm_local_mlp_0.1_kl_0.5_obj_0.1_grasp/checkpoints/best_eval_success_rate.pt"
        ),
        use_additional_motion_planner=True,
    ),
    # ------------------------------------- xarm xhand_right ------------------------------------------
    "xarm_xhand_grasp_generator": GraspGeneratorConfig(
        robot_config_name="xarm7_xhand_right",
        ms_robot_uid="xarm7_xhand_right",
        agent_config_name="spconv_layernorm_local_kin_loss",
        agent_checkpoint_path=str(
            S3_ROOT
            / "logs/train/xarm_xhand/ycb_objaverse_pinch_scenes_cvae_spconv_layernorm_local_mlp_0.1_kl_0.5_obj_0.1_grasp/checkpoints/iter_10000.pt"
        ),
    ),
}


class GraspGenerator:
    def __init__(self, cfg: GraspGeneratorConfig, device: Device):
        self.cfg = cfg
        self.device = device

        robot_config = build_robot_config(cfg.robot_config_name)
        agent_config: GraspPolicyConfig = GRASP_POLICY_CONFIGS[cfg.agent_config_name]
        self.agent = build_grasp_policy(agent_config, robot_config, device)
        ckpt = torch.load(ensure_local_filepath(cfg.agent_checkpoint_path), map_location=device)
        self.agent.load_state_dict(ckpt["agent"])
        self.agent.eval()

        grasp_optimizer_config = GraspOptimConfig()
        self.grasp_optimizer = WarpGraspOptimizer(grasp_optimizer_config, robot_config, device)
        # No ExperimentConfig here (robot built from the policy config), so no build_planning_traj_builder.
        self.traj_builder = GraspTrajBuilder(
            GraspTrajBuilderConfig(
                fixed_base_robot_config=robot_config,
                floating_base_robot_config=robot_config,  # policy-refined grasps planned as-is
                convert_to_fixed=False,
                motion_planner=MotionPlannerConfig(backend="torch"),
            ),
            device,
        )
        self.evaluator = build_env(
            EnvConfig(ms_env_config=MsEnvConfig(robot_uid=cfg.ms_robot_uid)), robot_config, device
        )

        if cfg.use_additional_motion_planner:
            additional_motion_planner_config = MotionPlannerConfig(backend="torch")
            self.additional_motion_planner = build_motion_planner(
                additional_motion_planner_config, robot_config, device=device
            )

    @torch.no_grad()
    def generate_grasps(
        self,
        scene_descriptions: List[SceneDescription],
        obs_batch_dict: Dict[str, torch.Tensor],
        num_samples_per_scene: int,
    ):
        actions = self.agent.get_action(obs_batch_dict, num_samples_per_scene=num_samples_per_scene)
        batch = GraspKitData(
            scene_ids=[d.scene_id for d in scene_descriptions],
            scene_descriptions=scene_descriptions,
        )
        batch.grasp_qpos = self.grasp_optimizer.optimize(
            batch.scene_descriptions, num_samples_per_scene, init_qpos=actions[..., 1, :]
        )
        batch, _ = self.traj_builder.build(batch)
        mp_success = batch.pre_traj_penetration < 0.0
        eval_results = self.evaluator.evaluate(batch)
        sim_success = eval_results["success"]
        success_mask = torch.logical_and(mp_success, sim_success)
        logger.info(
            f"Success rate: {success_mask.sum() / success_mask.numel():.2%}, MP: {mp_success.sum() / mp_success.numel():.2%}, Sim: {sim_success.sum() / sim_success.numel():.2%}"
        )

        if self.cfg.use_additional_motion_planner:
            additional_batch = self.additional_motion_planner.plan(batch, return_pre_post_only=True)
            additional_post_grasp_qpos = additional_batch.post_grasp_qpos
        return_list = []
        for scene_idx in range(len(scene_descriptions)):
            cur_dict = {
                "grasp_qpos": batch.grasp_qpos[scene_idx][success_mask[scene_idx]],
                "pre_grasp_qpos": batch.pre_grasp_qpos[scene_idx][success_mask[scene_idx]],
                "post_grasp_qpos": batch.post_grasp_qpos[scene_idx][success_mask[scene_idx]],
                "indices": torch.where(success_mask[scene_idx])[0],
            }
            if self.cfg.use_additional_motion_planner:
                cur_dict["additional_post_grasp_qpos"] = additional_post_grasp_qpos[scene_idx][success_mask[scene_idx]]
            return_list.append(cur_dict)
        return return_list


# ---------------------------------------------------------------------------
# Script config and main
# ---------------------------------------------------------------------------


@dataclass
class Config:
    skip_existing: bool = True
    """Skip existing scenes"""
    shuffle_order: bool = True
    """Shuffle the order of scenes"""

    scene_description_path: str = str(DATA_ROOT / "scene_description/ycb_objaverse/10_poses_1_scale.json")
    """Path to the scene description"""
    sim_data_path: str = str(S3_ROOT / "data/sim_data/ability_grasps/ycb_objaverse_7k_scenes_full_pc.h5")
    """Path to the h5 file containing the dataset"""
    keep_existing_actions: bool = True
    """Keep the original actions"""
    output_dir: str = str(S3_ROOT / "data/gen_data/ability_grasps/ycb_objaverse_7k_scenes/")
    """Output path"""

    num_points: int = 2048
    """Number of points to sample from the point cloud"""
    points_lower_bound: float = 0.01
    """Lower bound for the conditioned points"""
    num_samples_per_batch: int = 512
    """Number of samples per batch"""

    max_num_batches: int = 5
    """Maximum number of batches"""
    target_num_samples_per_scene: int = 512
    """Number of samples per scene"""

    grasp_generator_name: str = "xarm_ability_pointnet_grasp_generator"
    """Grasp generator name"""


def load_hdf5_data_to_memory(group):
    data = {}
    for key in group.keys():
        item = group[key]
        if hasattr(item, "keys"):
            data[key] = load_hdf5_data_to_memory(item)
        else:
            data[key] = item[()]
    return data


def generate_grasps(cfg: Config):
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    scene_ids = sorted(list(scene_descriptions.keys()))
    if cfg.shuffle_order:
        random.seed(int(time.time()) + int(os.getpid()))
        random.shuffle(scene_ids)

    seed_everything(42)
    logger.info(f"Num scenes: {len(scene_ids)}")

    if cfg.sim_data_path.endswith(".h5"):
        sim_data = h5py.File(ensure_local_filepath(cfg.sim_data_path), "r")
    grasp_generator_config = GRASP_GENERATOR_CONFIGS[cfg.grasp_generator_name]
    grasp_generator = GraspGenerator(grasp_generator_config, device=device)

    for scene_id in track(
        scene_ids,
        description="Generating",
        disable=True,
    ):
        output_path = os.path.join(cfg.output_dir, f"{scene_id}.h5")
        fs, _, _ = fsspec.get_fs_token_paths(output_path)
        if fs.exists(output_path) and cfg.skip_existing:
            logger.info(f"Skipping {scene_id} because its output already exists")
            continue

        if cfg.sim_data_path.endswith(".h5"):
            scene_data = sim_data["data"][scene_id]
        else:
            with fs.open(os.path.join(cfg.sim_data_path, f"{scene_id}.h5"), "rb") as f:
                with h5py.File(f, "r") as scene_file:
                    scene_data = load_hdf5_data_to_memory(scene_file)

        scene_pcd = scene_data["obs"]["pointcloud"][:]
        if scene_pcd.ndim == 3 and scene_pcd.shape[0] == 1:
            scene_pcd = np.squeeze(scene_pcd, axis=0)
        valid_mask = np.ones(scene_pcd.shape[0], dtype=bool)
        valid_mask = valid_mask & (scene_pcd[..., 2] > cfg.points_lower_bound)

        if "best_point_indices" in scene_data["obs"] and cfg.keep_existing_actions:
            best_point_indices = scene_data["obs"]["best_point_indices"][:]
            valid_mask[best_point_indices] = False

        if valid_mask.sum() == 0:
            logger.warning(f"No valid points found for {scene_id}")
            continue

        actions, num_actions = [], 0
        if "actions" in scene_data and cfg.keep_existing_actions:
            existing_actions = scene_data["actions"][:]
            existing_actions = {
                "grasp_qpos": torch.from_numpy(existing_actions[..., 0, :]),  # align with the output format
                "pre_grasp_qpos": torch.from_numpy(existing_actions[..., 1, :]),
                "post_grasp_qpos": torch.from_numpy(existing_actions[..., 2, :]),
                "additional_post_grasp_qpos": torch.from_numpy(existing_actions[..., 3, :]),
            }
            actions.append(existing_actions)
            num_actions = existing_actions["grasp_qpos"].shape[0]

        if not cfg.sim_data_path.endswith(".h5"):  # also try to download object assets from s3
            for mesh in scene_descriptions[scene_id].meshes.values():
                _ = ensure_local_filepath(os.path.join(S3_ROOT, mesh.collision_mesh_path))

        for batch_idx in range(cfg.max_num_batches):
            fps_indices = fpsample.bucket_fps_kdline_sampling(
                scene_pcd[valid_mask], min(cfg.num_samples_per_batch, valid_mask.sum()), h=3
            )
            cond_point_indices = np.where(valid_mask)[0][fps_indices]
            if len(cond_point_indices) < cfg.num_samples_per_batch:
                rand_cond_point_indices = np.random.choice(
                    np.setdiff1d(np.arange(scene_pcd.shape[0]), cond_point_indices),
                    size=cfg.num_samples_per_batch - len(cond_point_indices),
                    replace=False,
                )
                cond_point_indices = np.concatenate([cond_point_indices, rand_cond_point_indices])
            other_point_indices = np.random.choice(
                np.setdiff1d(np.arange(scene_pcd.shape[0]), cond_point_indices),
                size=cfg.num_points - cfg.num_samples_per_batch,
                replace=False,
            )
            point_indices = np.concatenate([cond_point_indices, other_point_indices], axis=0)
            obs_batch_dict = {
                "pointcloud": torch.from_numpy(scene_pcd[point_indices]).unsqueeze(0),
                "conditioned_point_indices": torch.arange(cfg.num_samples_per_batch, device=device).unsqueeze(0),
            }
            obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
            batch_actions = grasp_generator.generate_grasps(
                [scene_descriptions[scene_id]], obs_batch_dict, cfg.num_samples_per_batch
            )
            if len(batch_actions) != 1:
                raise ValueError(f"Expected 1 scene, got {len(batch_actions)}")
            succ_indices = cond_point_indices[batch_actions[0].pop("indices").numpy(force=True)]
            valid_mask[succ_indices] = False

            actions.append(batch_actions[0])
            num_actions += batch_actions[0]["grasp_qpos"].shape[0]
            logger.info(f"Batch {batch_idx} done, {num_actions} actions generated")

            if num_actions >= cfg.target_num_samples_per_scene:
                break

        if len(actions) == 0:
            logger.warning(f"No actions generated for {scene_id}")
            continue

        actions = {k: np.concatenate([a[k].numpy(force=True) for a in actions], axis=0) for k in actions[0].keys()}
        logger.info(f"Total {actions['grasp_qpos'].shape[0]} actions generated")

        mem_buffer = io.BytesIO()
        with h5py.File(mem_buffer, "w") as h5file:
            for key, value in actions.items():
                h5file.create_dataset(key, data=value)
        mem_buffer.seek(0)
        with fsspec.open(output_path, "wb") as f:
            f.write(mem_buffer.read())

    sim_data.close()


CONFIGS: Dict[str, Config] = {
    # ------------------------------------- shadow_right ----------------------------------------------
    "shadow_dex1b_v1": Config(
        grasp_generator_name="shadow_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/object_data/full_pointcloud/dex1b_v1.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/shadow_grasps/dex1b_v1/"),
        keep_existing_actions=False,
    ),
    # ------------------------------------- inspire_right ----------------------------------------------
    "inspire_dex1b_v1": Config(
        grasp_generator_name="inspire_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/object_data/full_pointcloud/dex1b_v1.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/inspire_grasps/dex1b_v1/"),
        keep_existing_actions=False,
    ),
    # ------------------------------------- ability_right ----------------------------------------------
    "ability_dex1b_v1": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dex1b_v1.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/ability_grasps/dex1b_v1/"),
        keep_existing_actions=False,
    ),
    "ability_dex1b_v1_x10": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        target_num_samples_per_scene=5120,
        max_num_batches=50,
        sim_data_path=str(S3_ROOT / "data/object_data/full_pointcloud/dex1b_v1.h5"),
        output_dir=str(S3_ROOT / "data/gen_data/ability_grasps/dex1b_v1_x10/"),
        keep_existing_actions=True,
    ),
    "ability_dex1b_v2": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v2.msgpack"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/object_data/full_pointcloud/dex1b_v2/"),
        output_dir=str(S3_ROOT / "data/gen_data/ability_grasps/dex1b_v2/"),
        keep_existing_actions=False,
    ),
    "ability_ycb_objaverse_7k_scenes": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/ycb_objaverse/7k_scenes.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/sim_data/ability_grasps/ycb_objaverse_7k_scenes_full_pc.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/ability_grasps/ycb_objaverse_7k_scenes/"),
        keep_existing_actions=False,
    ),
    "ability_ycb_objaverse_7k_scenes_new_actions": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/ycb_objaverse/7k_scenes.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/sim_data/ability_grasps/ycb_objaverse_7k_scenes_full_pc.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/ability_grasps/ycb_objaverse_7k_scenes_new_actions/"),
    ),
    # ---------------------- arti --------------------------------------
    "ability_partnet_trashcan_20_poses": Config(
        grasp_generator_name="ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/partnet/trashcan/20_candidates.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/sim_data/ability_arti/trashcan/20_poses_4_frames.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/ability_arti/partnet_trashcan_20_poses/"),
    ),
    # ------------------------------------- xarm ability_right ------------------------------------------
    "xarm_ability_dex1b_v1": Config(
        grasp_generator_name="xarm_ability_spconv_grasp_generator",
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        target_num_samples_per_scene=512,
        sim_data_path=str(S3_ROOT / "data/render_data/pointcloud/dex1b/v1_real_camera_04_21.h5"),
        output_dir=str(PROJECT_ROOT / "data/gen_data/xarm_ability_grasps/dex1b_v1/"),
        keep_existing_actions=False,
    ),
}


if __name__ == "__main__":
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    generate_grasps(cfg)
