# ruff: noqa: F821
"""Padded or packed multi-scene tensor payload with per-scene H5 persistence."""

import io
import os
from dataclasses import dataclass
from pathlib import Path

import fsspec
import h5py
import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor

from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import build_filesystem, ensure_local_filepath

_PERSISTED_PER_GRASP_FIELDS: tuple[str, ...] = (
    "pre_grasp_qpos",
    "grasp_qpos",
    "post_grasp_qpos",
)
_PER_GRASP_FIELDS: tuple[str, ...] = _PERSISTED_PER_GRASP_FIELDS + (
    "full_traj_qpos",
    "pre_traj_penetration",
)
_PER_SCENE_FIELDS: tuple[str, ...] = (
    "pointcloud",
    "pointcloud_rgb",
    "pointcloud_segmentation",
    "target_actor_segmentation_id",
    "camera_poses",
)
_PERSISTED_TENSOR_FIELDS: tuple[str, ...] = _PERSISTED_PER_GRASP_FIELDS + _PER_SCENE_FIELDS


@beartype
@dataclass
class GraspKitData:
    """Batched per-stage tensor payload; supports padded and packed layouts."""

    scene_ids: list[str]
    scene_descriptions: list[SceneDescription] | None = None

    # None → padded (B, K, ...). Set → packed (N, ...), N = first_indices[-1].
    first_indices: Int[Tensor, "Bp1"] | None = None

    # Per-grasp fields (padded or packed):
    pre_grasp_qpos: Float[Tensor, "B K dof"] | Float[Tensor, "N dof"] | None = None
    grasp_qpos: Float[Tensor, "B K dof"] | Float[Tensor, "N dof"] | None = None
    post_grasp_qpos: Float[Tensor, "B K dof"] | Float[Tensor, "N dof"] | None = None
    full_traj_qpos: Float[Tensor, "B K T dof"] | Float[Tensor, "N T dof"] | None = None
    pre_traj_penetration: Float[Tensor, "B K"] | Float[Tensor, "N"] | None = None

    # Per-scene fields (always (B, ...)):
    pointcloud: Float[Tensor, "B T Npts 3"] | None = None
    pointcloud_rgb: Float[Tensor, "B T Npts 3"] | None = None
    pointcloud_segmentation: Int[Tensor, "B T Npts 1"] | None = None
    target_actor_segmentation_id: Int[Tensor, "B"] | None = None
    camera_poses: Float[Tensor, "B T 7"] | None = None

    @property
    def num_scenes(self) -> int:
        return len(self.scene_ids)

    @property
    def is_packed(self) -> bool:
        return self.first_indices is not None

    @property
    def is_padded(self) -> bool:
        return self.first_indices is None

    @property
    def num_grasps_per_scene(self) -> Tensor:
        if self.first_indices is not None:
            return self.first_indices[1:] - self.first_indices[:-1]
        for name in _PER_GRASP_FIELDS:
            value = getattr(self, name)
            if value is not None:
                return torch.full((self.num_scenes,), value.shape[1], dtype=torch.long)
        return torch.zeros(self.num_scenes, dtype=torch.long)

    def to_padded(self) -> "GraspKitData":
        if self.is_padded:
            return self
        counts = self.num_grasps_per_scene
        if (counts != counts[0]).any():
            raise ValueError(f"to_padded requires uniform K across scenes, got {counts.tolist()}")
        num_grasps = int(counts[0].item())
        kwargs: dict[str, object] = {"scene_ids": self.scene_ids, "scene_descriptions": self.scene_descriptions}
        for name in _PER_GRASP_FIELDS:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value.reshape(self.num_scenes, num_grasps, *value.shape[1:])
        for name in _PER_SCENE_FIELDS:
            kwargs[name] = getattr(self, name)
        return GraspKitData(**kwargs)

    def to_packed(self) -> "GraspKitData":
        if self.is_packed:
            return self
        num_grasps = 0
        for name in _PER_GRASP_FIELDS:
            value = getattr(self, name)
            if value is not None:
                num_grasps = value.shape[1]
                break
        kwargs: dict[str, object] = {
            "scene_ids": self.scene_ids,
            "scene_descriptions": self.scene_descriptions,
            "first_indices": torch.arange(self.num_scenes + 1, dtype=torch.long) * num_grasps,
        }
        for name in _PER_GRASP_FIELDS:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value.reshape(self.num_scenes * num_grasps, *value.shape[2:])
        for name in _PER_SCENE_FIELDS:
            kwargs[name] = getattr(self, name)
        return GraspKitData(**kwargs)

    def filter(self, mask: Tensor) -> "GraspKitData":
        """Filter packed grasps or scenes by a boolean mask and drop empty scenes."""
        assert self.is_packed, "filter requires packed layout; call .to_packed() first."
        mask = mask.cpu()
        counts = self.num_grasps_per_scene
        n = int(self.first_indices[-1].item())
        if mask.shape == (self.num_scenes,):
            grasp_mask = mask.repeat_interleave(counts)
        else:
            assert mask.shape == (n,), f"expected mask shape ({n},) or ({self.num_scenes},), got {tuple(mask.shape)}"
            grasp_mask = mask
        scene_ids_per_grasp = torch.repeat_interleave(torch.arange(self.num_scenes), counts)
        new_counts = torch.zeros(self.num_scenes, dtype=torch.long).scatter_add_(
            0, scene_ids_per_grasp, grasp_mask.long()
        )
        scenes_kept = new_counts > 0
        kept_list = scenes_kept.tolist()
        kwargs: dict[str, object] = {
            "scene_ids": [sid for sid, k in zip(self.scene_ids, kept_list) if k],
            "scene_descriptions": (
                None
                if self.scene_descriptions is None
                else [d for d, k in zip(self.scene_descriptions, kept_list) if k]
            ),
            "first_indices": torch.cat([torch.zeros(1, dtype=torch.long), new_counts[scenes_kept].cumsum(0)]),
        }
        for name in _PER_GRASP_FIELDS:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value[grasp_mask.to(value.device)]
        for name in _PER_SCENE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value[scenes_kept.to(value.device)]
        return GraspKitData(**kwargs)

    def save(self, path: str, append: bool = False) -> None:
        """Save scenes to individual H5 files or groups in a merged H5 file."""
        per_scene: list[dict[str, Tensor]] = []
        for i in range(self.num_scenes):
            payload: dict[str, Tensor] = {}
            if self.first_indices is not None:
                start = int(self.first_indices[i].item())
                stop = int(self.first_indices[i + 1].item())
                for name in _PERSISTED_PER_GRASP_FIELDS:
                    value = getattr(self, name)
                    if value is not None:
                        payload[name] = value[start:stop]
            else:
                for name in _PERSISTED_PER_GRASP_FIELDS:
                    value = getattr(self, name)
                    if value is not None:
                        payload[name] = value[i]
            for name in _PER_SCENE_FIELDS:
                value = getattr(self, name)
                if value is not None:
                    payload[name] = value[i]
            per_scene.append(payload)

        if path.endswith(".h5"):
            self._write_h5(path, dict(zip(self.scene_ids, per_scene)), group_per_scene=True, append=append)
            return
        if not path.startswith("s3://"):
            os.makedirs(path, exist_ok=True)
        for scene_id, payload in zip(self.scene_ids, per_scene):
            self._write_h5(
                os.path.join(path, f"{scene_id}.h5"), {scene_id: payload}, group_per_scene=False, append=append
            )

    @staticmethod
    def _write_h5(
        path: str,
        scenes: dict[str, dict[str, Tensor]],
        group_per_scene: bool,
        append: bool,
    ) -> None:
        """Write local or remote H5 data, appending grasp rows and replacing scene fields."""
        buffer = io.BytesIO()
        if os.path.exists(path):
            with open(path, "rb") as f:
                buffer.write(f.read())
        buffer.seek(0)
        mode = "a" if buffer.getbuffer().nbytes > 0 else "w"
        with h5py.File(buffer, mode) as f:
            for scene_id, payload in scenes.items():
                group = f.require_group(scene_id) if group_per_scene else f
                write = {name: tensor.detach().cpu() for name, tensor in payload.items()}
                if append:
                    for name in _PERSISTED_PER_GRASP_FIELDS:
                        if name not in payload or name not in group or group[name].shape[0] == 0:
                            continue
                        old = torch.tensor(group[name][...])
                        new = payload[name].cpu()
                        write[name] = torch.cat([old, new], dim=0)
                for name, tensor in write.items():
                    if name in group:
                        del group[name]
                    group.create_dataset(name, data=tensor.numpy(force=True))
        buffer.seek(0)
        with fsspec.open(path, "wb") as out:
            out.write(buffer.read())

    @classmethod
    def load(
        cls,
        path: str,
        scene_ids: list[str] | None = None,
        packed: bool = True,
    ) -> "GraspKitData":
        """Load scenes in packed form or padded form with uniform grasp counts."""
        scene_payloads: dict[str, dict[str, Tensor]] = {}
        if path.endswith(".h5"):
            # Read remote files into memory and open local files in place.
            if path.startswith("s3://") or path.startswith("hf://"):
                handle = io.BytesIO(build_filesystem(path).cat_file(path))
            else:
                handle = path
            with h5py.File(handle, "r") as f:
                top = f.keys()
                if top and isinstance(f[next(iter(top))], h5py.Dataset):
                    # Single per-scene file: datasets live at the root, scene_id is the filename stem.
                    scene_id = Path(path).stem
                    scene_payloads[scene_id] = {n: torch.tensor(f[n][...]) for n in _PERSISTED_TENSOR_FIELDS if n in f}
                else:
                    for scene_id in scene_ids if scene_ids is not None else top:
                        if scene_id in f:
                            g = f[scene_id]
                            scene_payloads[scene_id] = {
                                n: torch.tensor(g[n][...]) for n in _PERSISTED_TENSOR_FIELDS if n in g
                            }
        else:
            local = ensure_local_filepath(path)
            if scene_ids is None:
                scene_ids = [p.stem for p in sorted(Path(local).glob("*.h5"))]
            for scene_id in scene_ids:
                h5_path = os.path.join(local, f"{scene_id}.h5")
                if not os.path.exists(h5_path):
                    continue
                with h5py.File(h5_path, "r") as f:
                    scene_payloads[scene_id] = {n: torch.tensor(f[n][...]) for n in _PERSISTED_TENSOR_FIELDS if n in f}

        if not scene_payloads:
            return cls(scene_ids=[])

        loaded_ids = list(scene_payloads.keys())
        per_scene = list(scene_payloads.values())
        grasp_counts = torch.tensor(
            [next((p[n].shape[0] for n in _PERSISTED_PER_GRASP_FIELDS if n in p), 0) for p in per_scene],
            dtype=torch.long,
        )
        kwargs: dict[str, object] = {"scene_ids": loaded_ids}
        if packed:
            kwargs["first_indices"] = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(grasp_counts, dim=0)])
            for name in _PERSISTED_PER_GRASP_FIELDS:
                present = [p[name] for p in per_scene if name in p]
                if len(present) == len(per_scene):
                    kwargs[name] = torch.cat(present, dim=0)
        else:
            if not bool((grasp_counts == grasp_counts[0]).all().item()):
                raise ValueError(f"load(packed=False) requires uniform K across scenes, got {grasp_counts.tolist()}")
            for name in _PERSISTED_PER_GRASP_FIELDS:
                present = [p[name] for p in per_scene if name in p]
                if len(present) == len(per_scene):
                    kwargs[name] = torch.stack(present, dim=0)
        for name in _PER_SCENE_FIELDS:
            present = [p[name] for p in per_scene if name in p]
            if len(present) == len(per_scene):
                kwargs[name] = torch.stack(present, dim=0)
        return cls(**kwargs)
