"""Viser replay format storing qpos, object poses, and asset paths."""

import io
from dataclasses import dataclass, field
from pathlib import Path

import fsspec
import h5py
import numpy as np
from beartype import beartype
from jaxtyping import Float

from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import GRASPKIT_HOME
from graspkit.scene.scene_description import SceneDescription


@beartype
@dataclass
class ReplayData:
    """One recorded trajectory for the replay viewer: robot qpos, object pose and the assets to draw them."""

    qpos: Float[np.ndarray, "T dof"]
    """FK variables: floating [xyz, euler_rxyz, joints] or fixed joints."""
    obj_pose: Float[np.ndarray, "T 7"]
    """Object world pose [xyz, quat_wxyz]."""
    robot_path: str
    """URDF or MJCF the viewer loads into `ViserUrdf` / `ViserMjcf`."""
    mesh_dir: str
    """URDF mesh directory; empty if none."""
    base_type: str
    """`"floating"` or `"fixed"`."""
    obj_mesh_path: str
    """Object collision mesh (loaded via trimesh)."""
    obj_scale: float
    stage: str = "sim-valid"
    """Viewer partition, e.g. optimize_grasps.py --verbose's "sample (unfiltered)" ... "sim"."""
    label: str = "success"
    """Free filter within a stage, e.g. `"success"` or `"failure"`."""
    # Fixed-base world pose; floating bases use qpos and leave these at identity.
    root_pos: Float[np.ndarray, "3"] = field(default_factory=lambda: np.zeros(3, np.float32))
    root_quat: Float[np.ndarray, "4"] = field(default_factory=lambda: np.array([1, 0, 0, 0], np.float32))
    # World-frame support table box (full size); the viewer draws it. Zero size = no table.
    table_size: Float[np.ndarray, "3"] = field(default_factory=lambda: np.zeros(3, np.float32))
    table_pos: Float[np.ndarray, "3"] = field(default_factory=lambda: np.zeros(3, np.float32))
    table_quat: Float[np.ndarray, "4"] = field(default_factory=lambda: np.array([1, 0, 0, 0], np.float32))

    @classmethod
    def from_traj(
        cls,
        qpos: Float[np.ndarray, "T dof"],
        obj_pose: Float[np.ndarray, "T 7"],
        robot_config: RobotConfig,
        scene_desc: SceneDescription,
        stage: str = "sim-valid",
        label: str = "success",
    ) -> "ReplayData":
        m = scene_desc.meshes[scene_desc.target_actor_id]
        fixed = robot_config.base_type == "fixed"
        tb = scene_desc.boxes["table"] if "table" in scene_desc.boxes else None
        return cls(
            qpos=np.asarray(qpos, dtype=np.float32),
            obj_pose=np.asarray(obj_pose, dtype=np.float32),
            robot_path=robot_config.urdf_or_mjcf_path,
            mesh_dir=robot_config.mesh_dir or "",
            base_type=robot_config.base_type,
            obj_mesh_path=str(Path(GRASPKIT_HOME) / m.collision_mesh_path),
            obj_scale=float(m.scales[0]),
            stage=stage,
            label=label,
            root_pos=np.asarray(robot_config.init_position if fixed else (0, 0, 0), dtype=np.float32),
            root_quat=np.asarray(robot_config.init_quaternion if fixed else (1, 0, 0, 0), dtype=np.float32),
            table_size=np.asarray(tb.size if tb else (0, 0, 0), dtype=np.float32),
            table_pos=np.asarray(tb.position if tb else (0, 0, 0), dtype=np.float32),
            table_quat=np.asarray(tb.quaternion if tb else (1, 0, 0, 0), dtype=np.float32),
        )

    @staticmethod
    def save(replays: list["ReplayData"], path: str) -> None:
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as f:
            for i, r in enumerate(replays):
                g = f.create_group(str(i))
                g.create_dataset("qpos", data=np.asarray(r.qpos, dtype=np.float32))
                g.create_dataset("obj_pose", data=np.asarray(r.obj_pose, dtype=np.float32))
                g.attrs["robot_path"] = r.robot_path
                g.attrs["mesh_dir"] = r.mesh_dir
                g.attrs["base_type"] = r.base_type
                g.attrs["obj_mesh_path"] = r.obj_mesh_path
                g.attrs["obj_scale"] = r.obj_scale
                g.attrs["stage"] = r.stage
                g.attrs["label"] = r.label
                g.attrs["root_pos"] = np.asarray(r.root_pos, dtype=np.float32)
                g.attrs["root_quat"] = np.asarray(r.root_quat, dtype=np.float32)
                g.attrs["table_size"] = np.asarray(r.table_size, dtype=np.float32)
                g.attrs["table_pos"] = np.asarray(r.table_pos, dtype=np.float32)
                g.attrs["table_quat"] = np.asarray(r.table_quat, dtype=np.float32)
        buffer.seek(0)
        with fsspec.open(path, "wb") as out:
            out.write(buffer.read())

    @staticmethod
    def load(path: str) -> list["ReplayData"]:
        with fsspec.open(path, "rb") as fp:
            data = fp.read()
        replays: list[ReplayData] = []
        with h5py.File(io.BytesIO(data), "r") as f:
            for key in sorted(f.keys(), key=int):
                g = f[key]
                replays.append(
                    ReplayData(
                        qpos=g["qpos"][...],
                        obj_pose=g["obj_pose"][...],
                        robot_path=str(g.attrs["robot_path"]),
                        mesh_dir=str(g.attrs["mesh_dir"]),
                        base_type=str(g.attrs["base_type"]),
                        obj_mesh_path=str(g.attrs["obj_mesh_path"]),
                        obj_scale=float(g.attrs["obj_scale"]),
                        stage=str(g.attrs["stage"]),
                        label=str(g.attrs["label"]),
                        root_pos=np.asarray(g.attrs["root_pos"], dtype=np.float32),
                        root_quat=np.asarray(g.attrs["root_quat"], dtype=np.float32),
                        table_size=np.asarray(g.attrs["table_size"], dtype=np.float32),
                        table_pos=np.asarray(g.attrs["table_pos"], dtype=np.float32),
                        table_quat=np.asarray(g.attrs["table_quat"], dtype=np.float32),
                    )
                )
        return replays
