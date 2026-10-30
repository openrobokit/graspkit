from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.scene.scene_description import SceneDescription


@dataclass
class Rollout:
    """Raw per-frame trajectory a sim evaluator records; a pipeline builder turns it into `ReplayData`."""

    qpos: np.ndarray  # (T, dof)
    obj_pose: np.ndarray  # (T, 7)
    scene_id: str
    success: bool = True
    grasp_index: int = 0  # which grasp of the scene, so replays key back to the same grasp across stages


class BaseEnv(ABC):
    """Unified grasp env: `MsEnv` (ManiSkill), `MjEnv` (MuJoCo), `IsaacEnv` (Isaac), `RealEnv` (hardware).
    - `evaluate` is the batched-rollout, reported-SR path shared by all eval call sites.
    - `reset`/`get_qpos`/`step_qpos`/`get_rgb`/`get_pointcloud` are the deploy primitives shared by sim and real.
    - `get_object_pose`/`set_object_pose`/`observe` drive simulated closed-loop policy evaluation.
    """

    def __init__(self):
        self.last_rollouts: list[Rollout] = []  # populated by evaluate(record=True)

    @abstractmethod
    def reset(
        self,
        scene: SceneDescription | list[SceneDescription] | None = None,
        init_qpos: torch.Tensor | None = None,
        root_pose: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ):
        """Initialize the scene (simulators accept a list, one env each), robot, and calibrated camera pose."""

    @abstractmethod
    def get_qpos(self) -> torch.Tensor:
        """Current joint positions, ours order, shape (B, D)."""

    @abstractmethod
    def step_qpos(self, q: torch.Tensor):
        """Command target `q` (ours order) and advance one control step."""

    @abstractmethod
    def get_info(self) -> dict[str, torch.Tensor]:
        """Return backend metrics for the current state."""

    def set_qpos_traj(
        self,
        traj_qpos: torch.Tensor,
        qpos_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stream one `(T, D)` qpos trajectory through the backend; the callback sees measured qpos."""
        assert traj_qpos.shape[0] > 0
        for step, qpos in enumerate(traj_qpos):
            self.step_qpos(qpos)
            if qpos_callback is not None:
                qpos_callback(step, self.get_qpos()[0])
        return self.get_info()

    @abstractmethod
    def get_rgb(self) -> np.ndarray:
        """Latest RGB camera image, shape (H, W, 3), uint8."""

    @abstractmethod
    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        """Object cloud selected by an aligned image mask, in world frame, shape (B, N, 3)."""

    def get_object_pose(self) -> torch.Tensor:
        """Target object world poses, shape (B, 7) [xyz, quat_wxyz]; simulators only."""
        raise NotImplementedError

    def set_object_pose(self, pose: torch.Tensor):
        """Teleport the target objects to world poses (B, 7); simulators only."""
        raise NotImplementedError

    def observe(self) -> dict[str, torch.Tensor]:
        """Sim-only, robot-free: `rgb` (B, V, H, W, 3), world `xyz`/`seg` (B, N, 3/1), `seg_id`, `camera_poses`."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        """Rollout of `batch.full_traj_qpos`; infos keyed by metric; `record=True` fills `self.last_rollouts`."""

    @abstractmethod
    def close(self):
        """Release the backend (sim envs / worker pool / hardware)."""
