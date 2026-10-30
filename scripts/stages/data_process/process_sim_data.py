"""
Usage:
uv run scripts/stages/data_process/process_sim_data.py g1_inspire_dex1b_tabletop_mini
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from rich.progress import track
from torch.nn import functional as F

from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.stages.contact_extractor import ContactExtractor
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.stage_utils import write_completion_marker

logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)
PROCESS_COMPLETE_MARKER = ".process_sim_data_complete"


def update_stats(
    batch: np.ndarray, count: int, mean: Optional[np.ndarray], M2: Optional[np.ndarray]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Update running statistics with a batch of data using Welford's algorithm."""
    n = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_M2 = ((batch - batch_mean) ** 2).sum(axis=0)
    if count == 0 or mean is None or M2 is None:
        return n, batch_mean, batch_M2
    new_count = count + n
    delta = batch_mean - mean
    new_mean = (count * mean + n * batch_mean) / new_count
    new_M2 = M2 + batch_M2 + (delta**2) * (count * n / new_count)
    return new_count, new_mean, new_M2


def create_dataset(exp_cfg: ExperimentConfig):
    os.makedirs(os.path.dirname(exp_cfg.processed_data_path), exist_ok=True)
    action_data_path = ensure_local_filepath(exp_cfg.grasp_output_dir)
    action_is_dir = os.path.isdir(action_data_path)
    native_actions = None
    if Path(action_data_path).suffix == ".pt":
        assert not exp_cfg.enable_contact_extraction
        replay = torch.load(action_data_path, map_location="cpu", weights_only=True)
        assert len(replay["scene_ids"]) == len(set(replay["scene_ids"]))
        assert replay["actions"].ndim == 3 and replay["actions"].shape[:2] == replay["rewards"].shape
        native_actions = {
            sid: actions[rewards > 0.5]
            .reshape(-1, exp_cfg.agent.num_queries, actions.shape[-1] // exp_cfg.agent.num_queries)
            .numpy()
            for sid, actions, rewards in zip(replay["scene_ids"], replay["actions"], replay["rewards"], strict=True)
        }
        del replay
    action_file = h5py.File(action_data_path, "r") if not action_is_dir and native_actions is None else None

    rendered_dirs = [ensure_local_filepath(exp_cfg.rendered_data_path)]
    logger.info(f"Rendered data sources: {rendered_dirs}")

    output_f = h5py.File(exp_cfg.processed_data_path, "w")
    data_group = output_f.create_group("data")
    meta_group = output_f.create_group("meta")

    if exp_cfg.enable_contact_extraction:
        contact_extractor = ContactExtractor(exp_cfg.robot_config, device)

    count_actions: int = 0
    mean_actions: Optional[np.ndarray] = None
    M2_actions: Optional[np.ndarray] = None
    seen_scene_ids: set[str] = set()

    for rendered_dir in rendered_dirs:
        if os.path.isdir(rendered_dir):
            scene_ids = sorted(p.stem for p in Path(rendered_dir).glob("*.h5"))
        else:
            with h5py.File(rendered_dir, "r") as f:
                scene_ids = sorted(f["data"] if native_actions is not None else f.keys())
        scene_ids = [sid for sid in scene_ids if sid not in seen_scene_ids]
        logger.info(f"Processing {len(scene_ids)} scenes from {rendered_dir}")
        for scene_id in track(scene_ids, description=f"Processing {os.path.basename(rendered_dir)}", disable=False):
            if native_actions is not None:
                actions = native_actions[scene_id]
                if not len(actions):
                    continue
                with h5py.File(rendered_dir, "r") as rendered:
                    source = rendered[f"data/{scene_id}/obs"]
                    group = data_group.create_group(scene_id)
                    obs = group.create_group("obs")
                    for key in source:
                        if key != "rgb":
                            source.copy(key, obs)
                    images = torch.from_numpy(source["rgb"][:]).permute(0, 3, 1, 2).float()
                    images = F.interpolate(images, size=(128, 128), mode="bilinear", align_corners=False)
                    obs.create_dataset(
                        "rgb",
                        data=images.round().byte().permute(0, 2, 3, 1).numpy(),
                        compression="gzip",
                        compression_opts=4,
                    )
                group.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)
                count_actions, mean_actions, M2_actions = update_stats(actions, count_actions, mean_actions, M2_actions)
                seen_scene_ids.add(scene_id)
                continue
            obs_data = GraspKitData.load(rendered_dir, scene_ids=[scene_id], packed=False)
            if obs_data.pointcloud is None:
                continue
            pointcloud = obs_data.pointcloud[0].numpy()  # (V, Npts, 3) — V views (1 for fixed mode)
            pointcloud_seg = obs_data.pointcloud_segmentation[0].numpy()  # (V, Npts, 1)
            if exp_cfg.renderer.camera_mode == "fixed":
                num_views = len(torch.unique(obs_data.camera_poses[0], dim=0))
                pointcloud = pointcloud.reshape(-1, num_views * pointcloud.shape[-2], 3)
                pointcloud_seg = pointcloud_seg.reshape(*pointcloud.shape[:-1], 1)
            scene_points = pointcloud[0]  # first view (contact extraction / target check use one view)
            scene_points_seg = pointcloud_seg[0]
            tgt_obj_seg_id = int(obs_data.target_actor_segmentation_id[0].item())
            tgt_obj_pts = scene_points[np.squeeze(scene_points_seg, axis=-1) == tgt_obj_seg_id]
            if tgt_obj_pts.shape[0] == 0:  # NOTE: i don't know why this happens
                logger.warning(f"No target object points found for scene {scene_id}")
                continue

            # Load action data (per-scene h5 dir, or single h5 keyed by scene_id); shared across attempts.
            if action_is_dir:
                if not os.path.exists(os.path.join(action_data_path, f"{scene_id}.h5")):
                    continue
                action = GraspKitData.load(action_data_path, scene_ids=[scene_id], packed=True)
            elif scene_id in action_file:
                action = GraspKitData.load(action_data_path, scene_ids=[scene_id], packed=True)
            else:
                continue

            data_group.create_group(scene_id)
            obs_group = data_group[scene_id].create_group("obs")
            obs_path = Path(rendered_dir) / f"{scene_id}.h5" if os.path.isdir(rendered_dir) else rendered_dir
            with h5py.File(obs_path, "r") as rendered:
                source = rendered if os.path.isdir(rendered_dir) else rendered[scene_id]
                if "rgb" in source:
                    source.copy("rgb", obs_group)
            obs_group.create_dataset("pointcloud", data=pointcloud, compression="gzip", compression_opts=4)
            if obs_data.pointcloud_rgb is not None:
                obs_group.create_dataset(
                    "pointcloud_rgb",
                    data=obs_data.pointcloud_rgb[0].numpy().reshape(pointcloud.shape),
                    compression="gzip",
                    compression_opts=4,
                )
            obs_group.create_dataset(
                "pointcloud_segmentation", data=pointcloud_seg, compression="gzip", compression_opts=4
            )
            obs_group.create_dataset("target_actor_segmentation_id", data=np.array(tgt_obj_seg_id))
            if obs_data.camera_poses is not None:  # (V, 7) per-view camera poses, for perception augmentation
                obs_group.create_dataset("camera_poses", data=obs_data.camera_poses[0].numpy().reshape(-1, 7))

            actions = np.stack(
                [
                    action.pre_grasp_qpos.numpy(),
                    action.grasp_qpos.numpy(),
                    action.post_grasp_qpos.numpy(),
                ],
                axis=-2,
            )
            grasp_jv = action.grasp_qpos.numpy()
            if exp_cfg.enable_contact_extraction:
                extracted = contact_extractor.extract_contacts(
                    torch.from_numpy(tgt_obj_pts).float().to(device),
                    torch.from_numpy(grasp_jv).float().to(device),
                )
                if extracted is None:
                    logger.warning(f"Contact extraction failed for scene {scene_id}, skipping")
                    del data_group[scene_id]
                    continue
                graspness_weights, action_indices, best_point_indices = extracted
                obs_group.create_dataset(
                    "graspness_weights",
                    data=graspness_weights.numpy(force=True),
                    compression="gzip",
                    compression_opts=4,
                )
                obs_group.create_dataset(
                    "action_indices",
                    data=action_indices.numpy(force=True),
                    compression="gzip",
                    compression_opts=4,
                )
                obs_group.create_dataset(
                    "best_point_indices",
                    data=best_point_indices.numpy(force=True),
                    compression="gzip",
                    compression_opts=4,
                )
            count_actions, mean_actions, M2_actions = update_stats(actions, count_actions, mean_actions, M2_actions)
            data_group[scene_id].create_dataset("actions", data=actions, compression="gzip", compression_opts=4)
            seen_scene_ids.add(scene_id)
    logger.info(f"Total number of actions: {count_actions}")
    assert count_actions > 0, f"No scenes produced actions in {exp_cfg.processed_data_path}"
    action_std = np.sqrt(M2_actions / count_actions)
    meta_group.create_dataset("action_mean", data=mean_actions)
    meta_group.create_dataset("action_std", data=action_std)
    output_f.close()
    if action_file is not None:
        action_file.close()
    logger.info(f"Processed data saved to {exp_cfg.processed_data_path}")
    write_completion_marker(exp_cfg.processed_data_path + PROCESS_COMPLETE_MARKER)


if __name__ == "__main__":
    from graspkit.configs.experiment import cli_experiment

    exp_cfg: ExperimentConfig = cli_experiment()
    create_dataset(exp_cfg)
