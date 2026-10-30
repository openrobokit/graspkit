"""HDF5 `data/<scene>/{obs,actions}`, optional `meta/action_{mean,std}`; prepared views are centered and pre-sampled."""

from typing import Literal

import h5py
import numpy as np
import torch
from robokit.xform.numpy import axis_angle_to_matrix
from torch.utils.data import Dataset

from graspkit.configs.experiment_config import DatasetConfig
from graspkit.datasets import perception_aug
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import ensure_local_filepath


class GraspDataset(Dataset):
    """Per-scene point-cloud observations and successful grasp trajectories from a processed HDF5."""

    def __init__(
        self,
        cfg: DatasetConfig,
        h5_path: str,
        scene_description_path: str,
        split: Literal["train", "test"] | None = "train",
        indices_only: bool = False,
    ):
        self.cfg, self.indices_only = cfg, indices_only
        descriptions = SceneDescription.load_scene_descriptions_msgpack(
            scene_description_path, return_splits=split is not None
        )
        self.scene_descriptions = descriptions if split is None else descriptions[split]
        self.h5_file = h5py.File(ensure_local_filepath(h5_path), "r")
        self.data = self.h5_file["data"]
        self.scene_ids = sorted(
            sid
            for sid in set(self.data) & set(self.scene_descriptions)
            if not cfg.return_actions or "actions" in self.data[sid]
        )
        self.sizes = {
            sid: (len(self.data[sid]["obs/pointcloud"]), len(self.data[sid]["actions"]) if cfg.return_actions else 0)
            for sid in self.scene_ids
        }
        assert not cfg.return_graspness or cfg.return_actions
        if indices_only:
            assert cfg.return_actions and not cfg.return_graspness and not cfg.return_rgb and not cfg.fuse_views
            counts = [self.sizes[sid][1] for sid in self.scene_ids]
            self.offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        elif cfg.cache_in_memory:
            cached = {}
            for sid in self.scene_ids:
                group = self.data[sid]
                obs = {
                    key: value[()]
                    for key, value in group["obs"].items()
                    if cfg.return_rgb or key not in ("rgb", "camera_poses")
                }
                if cfg.return_object_pointcloud_only and not cfg.return_graspness and not cfg.fuse_views:
                    masks = obs["pointcloud_segmentation"][..., 0] == obs["target_actor_segmentation_id"]
                    for key in ("pointcloud", "pointcloud_segmentation"):
                        obs[key] = [view[mask] for view, mask in zip(obs[key], masks)]
                cached[sid] = {"obs": obs}
                if cfg.return_actions:
                    cached[sid]["actions"] = group["actions"][:]
            self.data = cached

    def __len__(self) -> int:
        return len(self.scene_ids)

    def to_device(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        """Load prepared observations and actions once for CUDA batch gathering."""
        assert self.indices_only and self.cfg.return_object_pointcloud_only
        lookup, views, render, actions = {}, [], [], []
        for sid in self.scene_ids:
            group = self.h5_file["data"][sid]
            obs = group["obs"]
            assert "pointcloud_center" in obs, "CUDA batching requires prepared HDF5 views"
            key = obs.id
            if key not in lookup:
                target = int(obs["target_actor_segmentation_id"][()])
                assert np.all(obs["pointcloud_segmentation"][:] == target)
                lookup[key] = len(views)
                views.append(obs)
            render.append(lookup[key])
            actions.append(group["actions"][:])
        points = np.stack([obs["pointcloud"][:] for obs in views])
        assert points.shape[2] == self.cfg.num_points
        arrays = {
            "points": points,
            "center": np.stack([obs["pointcloud_center"][:] for obs in views]),
            "target_id": np.array([obs["target_actor_segmentation_id"][()] for obs in views], dtype=np.int32),
            "render": np.array(render),
            "flat_actions": np.concatenate(actions),
        }
        return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}

    def __getitem__(self, idx: int) -> dict:
        scene_id = self.scene_ids[idx]
        views, count = self.sizes[scene_id]
        view_idx = 0 if self.cfg.fuse_views else np.random.randint(views)
        result = {"metas": {"scene_id": scene_id}}
        if self.indices_only:
            action_indices = np.random.choice(count, self.cfg.num_actions, replace=count < self.cfg.num_actions)
            return {"entry": idx, "draw": view_idx, "rows": torch.from_numpy(action_indices + self.offsets[idx])}

        scene_data = self.data[scene_id]
        obs = scene_data["obs"]
        pcd_views = obs["pointcloud"]
        scene_pcd = pcd_views[view_idx].astype(np.float32, copy=False)
        scene_pcd_seg = obs["pointcloud_segmentation"][view_idx]
        target = int(obs["target_actor_segmentation_id"][()])
        if self.cfg.fuse_views:
            cfg = self.cfg
            assert cfg.return_object_pointcloud_only and not cfg.return_graspness and "pointcloud_center" not in obs
            points = np.asarray(pcd_views, dtype=np.float32)
            segments = np.asarray(obs["pointcloud_segmentation"])
            if cfg.aug_prob > 0 and np.random.rand() < cfg.aug_prob:
                rng = np.random.default_rng(np.random.randint(1 << 31))
                scene_pcd = perception_aug.fuse(
                    points, segments[..., 0], target, obs["camera_poses"][:], rng, True,
                    max_wrist=cfg.fuse_views, drop_wrist_prob=cfg.aug_drop_wrist_prob,
                    rgb_prob=cfg.aug_rgb_prob, keep_min=cfg.aug_keep_min, voxel=cfg.aug_voxel_max,
                    view_noise=cfg.view_noise,
                )  # fmt: skip
                if len(scene_pcd) == 0:
                    scene_pcd = points[0][segments[0, :, 0] == target]
            else:
                others = np.random.permutation(np.arange(1, views))
                selected = [0, *others[: np.random.randint(min(cfg.fuse_views, len(others)) + 1)]]
                clouds = []
                for view in selected:
                    pts = points[view][segments[view, :, 0] == target]
                    rot = axis_angle_to_matrix(np.random.randn(3) * cfg.view_noise * 10)
                    center = pts.mean(0) if len(pts) else np.zeros(3)
                    clouds.append((pts - center) @ rot.T + center + np.random.randn(3) * cfg.view_noise)
                scene_pcd = np.concatenate(clouds).astype(np.float32)
            scene_pcd = scene_pcd + np.random.randn(*scene_pcd.shape).astype(np.float32) * cfg.point_noise
            scene_pcd_seg = np.full((len(scene_pcd), 1), target, dtype=segments.dtype)
        if self.cfg.return_actions:
            action_indices = np.random.choice(count, self.cfg.num_actions, replace=count < self.cfg.num_actions)
            result["actions"] = torch.from_numpy(scene_data["actions"][:][action_indices])
        object_indices = np.flatnonzero(scene_pcd_seg[:, 0] == target)
        if "pointcloud_center" in obs:
            assert len(scene_pcd) == self.cfg.num_points and not self.cfg.return_graspness
            assert np.all(scene_pcd_seg == target)
            scene_pcd = scene_pcd + obs["pointcloud_center"][view_idx]
            observations = {"pointcloud": torch.from_numpy(scene_pcd), "objectness": torch.from_numpy(scene_pcd_seg)}
        elif not self.cfg.return_graspness:
            candidates = object_indices if self.cfg.return_object_pointcloud_only else np.arange(len(scene_pcd))
            point_indices = np.random.choice(
                candidates, self.cfg.num_points, replace=len(candidates) <= self.cfg.num_points
            )
            observations = {
                "pointcloud": torch.from_numpy(scene_pcd[point_indices]),
                "objectness": torch.from_numpy(scene_pcd_seg[point_indices]),
            }
        else:
            weights = obs["graspness_weights"][:]
            full_objectness = scene_pcd_seg[:, 0] == target
            object_points = scene_pcd[full_objectness]
            best_indices = obs["best_point_indices"][:][action_indices]
            point_actions = obs["action_indices"][:]
            for i in range(len(best_indices)):
                if np.random.rand() < 0.5:
                    close = np.flatnonzero(point_actions == action_indices[i])
                    if len(close):
                        best_indices[i] = np.random.choice(close, size=1, replace=False)[0]
            best_points = object_points[best_indices]
            remaining = self.cfg.num_points - len(best_points)
            candidates = object_indices if self.cfg.return_object_pointcloud_only else np.arange(len(scene_pcd))
            other_indices = np.random.choice(
                candidates, remaining, replace=self.cfg.return_object_pointcloud_only and len(candidates) <= remaining
            )
            full_weights = np.zeros(len(scene_pcd), dtype=np.float32)
            full_weights[full_objectness] = weights
            observations = {
                "pointcloud": torch.from_numpy(np.concatenate([best_points, scene_pcd[other_indices]])),
                "objectness": torch.from_numpy(
                    np.concatenate([np.ones(len(best_points), dtype=bool), full_objectness[other_indices]])
                ).float(),
                "graspness": torch.from_numpy(
                    np.concatenate([weights[best_indices], full_weights[other_indices]])
                ).float(),
                "best_point_indices": torch.arange(len(best_points)),
            }
        if self.cfg.return_rgb and "rgb" in obs:
            cameras = len(obs["rgb"]) // len(pcd_views)
            images = obs["rgb"][view_idx * cameras : (view_idx + 1) * cameras]
            observations["rgb"] = torch.from_numpy(images).permute(0, 3, 1, 2).flatten(0, 1)
            observations["qpos"] = torch.from_numpy(obs["qpos"][:]).float()
            observations["camera_poses"] = torch.from_numpy(obs["camera_poses"][:]).float()
        elif "qpos" in obs:
            observations["qpos"] = torch.from_numpy(obs["qpos"][:]).float()
        return {**result, "observations": observations}
