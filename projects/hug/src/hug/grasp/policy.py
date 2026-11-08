"""The CVAE grasp policy: ONNX load, inference, and the one stage-decode convention.

Input ``pointcloud`` ``(1, 1024, 3)`` in W0 plus a latent ``z`` ``(1, 1, latent_dim)``; output ``(3, 13)``: the
``R_hand_base_link`` target at pre-grasp, grasp and post-grasp, each ``[pos(3), quat_wxyz(4), hand_6(6)]``, the hand
in URDF order (thumb rotate, thumb bend, index, middle, ring, pinky). Sim and real both decode via :func:`parse_stages`.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from hug import paths

# The policy's idle hand (thumb pitch -0.4, the training convention); stage 0 is forced to it.
POLICY_IDLE_HAND = np.array([1.3, -0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def parse_stages(stages: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``(3, 13)`` policy output → ``(stage_poses, stage_hands)`` as float32 lists.

    Stage 0's thumb → ``POLICY_IDLE_HAND[:2]``; stage 2's pose = stage 1's (POST_GRASP is hand-only)."""
    assert stages.shape == (3, 13), f"expected 3x13, got {stages.shape}"
    stage_poses, stage_hands = [], []
    for i in range(3):
        pos = stages[i, :3].astype(np.float32).copy()
        quat_wxyz = stages[i, 3:7].astype(np.float32)
        hand_6 = stages[i, 7:].astype(np.float32)
        if i == 0:
            hand_6[:2] = POLICY_IDLE_HAND[:2]
        stage_poses.append(np.concatenate([pos, quat_wxyz]).astype(np.float32))
        stage_hands.append(hand_6)
    stage_poses[2] = stage_poses[1].copy()
    return stage_poses, stage_hands


class GraspPolicy:
    """ONNX CVAE grasp policy: object cloud in W0 → three stages of ``(pose7, hand_6)``.

    Latent ``z = 0`` by default: same cloud, same grasp. ``sample=True`` draws ``z`` from ``np.random``."""

    def __init__(self, path: Optional[Path] = None) -> None:
        import onnxruntime as ort

        path = path if path is not None else paths.DEFAULT_GRASP_POLICY
        if path.suffix.lower() != ".onnx":
            raise ValueError(f"unsupported policy extension: {path}")
        self.path = path
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail] or [
            "CPUExecutionProvider"
        ]
        self._sess = ort.InferenceSession(str(path), providers=providers)

        in_specs = {i.name: i for i in self._sess.get_inputs()}
        if set(in_specs) != {"pointcloud", "z"}:
            raise RuntimeError(f"expected a CVAE graph with inputs ('pointcloud', 'z'), got {sorted(in_specs)}")
        n_dim, z_dim = in_specs["pointcloud"].shape[1], in_specs["z"].shape[2]
        if not isinstance(n_dim, int) or not isinstance(z_dim, int):
            raise RuntimeError(f"ONNX N / latent axes must be static, got {n_dim!r} / {z_dim!r}")
        self.num_points = int(n_dim)
        self.latent_dim = int(z_dim)
        self._out_name = self._sess.get_outputs()[0].name

    def infer(self, sampled_pc: np.ndarray, *, sample: bool = False) -> np.ndarray:
        """One forward pass on an ``(num_points, 3)`` cloud → raw ``(3, 13)`` stages."""
        if sample:
            z = np.random.standard_normal((1, 1, self.latent_dim)).astype(np.float32)
        else:
            z = np.zeros((1, 1, self.latent_dim), dtype=np.float32)
        feeds = {"pointcloud": sampled_pc[None].astype(np.float32), "z": z}
        actions = self._sess.run([self._out_name], feeds)[0]
        while actions.ndim > 2:  # exports differ in leading singleton axes
            actions = actions[0]
        return actions

    def predict(self, pc_world: np.ndarray, *, sample: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Subsample ``pc_world`` to ``num_points`` (``np.random``), infer, and decode via :func:`parse_stages`."""
        n = pc_world.shape[0]
        idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
        stages = self.infer(pc_world[idx], sample=sample)
        assert stages.shape == (3, 13), f"expected 3x13, got {stages.shape}"
        return parse_stages(stages)
