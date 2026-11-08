"""GR00T WholeBodyControl lower-body policy: standing and walking ONNX models, dispatched on command magnitude.

Mirrors NVIDIA's GR00T WBC reference inference (29-DoF observation, 15-DoF action) minus its node scaffolding.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from hug import paths
from hug.utils import LPFilter

# fmt: off
DEFAULT_LOWER_QPOS: NDArray[np.float32] = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,    # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,    # right leg
     0.0, 0.0, 0.0,                     # waist (yaw, roll, pitch)
], dtype=np.float32)

# PD gains for the 15 lower joints (GR00T g1_gear_wbc.yaml); sim and real both use them.
DEFAULT_LOWER_KP: NDArray[np.float64] = np.array([
    150.0, 150.0, 150.0, 200.0, 40.0, 40.0,    # left leg
    150.0, 150.0, 150.0, 200.0, 40.0, 40.0,    # right leg
    400.0, 400.0, 400.0,                        # waist
])
DEFAULT_LOWER_KD: NDArray[np.float64] = np.array([
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    10.0, 10.0, 10.0,
])
# fmt: on


def _quat_rotate_inverse(q_wxyz: NDArray[np.float32], v: NDArray[np.float32]) -> NDArray[np.float32]:
    w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
    cw, cx, cy, cz = w, -x, -y, -z
    return np.array(
        [
            v[0] * (cw**2 + cx**2 - cy**2 - cz**2) + v[1] * 2 * (cx * cy - cw * cz) + v[2] * 2 * (cx * cz + cw * cy),
            v[0] * 2 * (cx * cy + cw * cz) + v[1] * (cw**2 - cx**2 + cy**2 - cz**2) + v[2] * 2 * (cy * cz - cw * cx),
            v[0] * 2 * (cx * cz - cw * cy) + v[1] * 2 * (cy * cz + cw * cx) + v[2] * (cw**2 - cx**2 - cy**2 + cz**2),
        ],
        dtype=np.float32,
    )


class LowerBodyPolicy:
    """Dual-policy GR00T WBC lower-body controller: 29-DoF observation → 15-DoF ``qdes``."""

    # fmt: off
    HEIGHT_BASE  = 0.74
    N_OBS        = 86
    N_HISTORY    = 6
    N_ACTIONS    = 15

    ACTION_SCALE        = 0.25
    ANG_VEL_SCALE       = 0.5
    DOF_POS_SCALE       = 1.0
    DOF_VEL_SCALE       = 0.05
    CMD_SCALE: NDArray[np.float32] = np.array([0.6, 0.6, 0.5], dtype=np.float32)
    CMD_THRESHOLD       = 0.05   # below this magnitude → use standing policy
    ROOT_VEL_FILTER_ALPHA = 0.3
    WARMUP_STEPS        = 100    # 2 s @ 50 Hz; blend default → policy on activation
    # fmt: on

    def __init__(self, standing_ckpt: Optional[Path] = None, walking_ckpt: Optional[Path] = None) -> None:
        self._standing = ort.InferenceSession(str(standing_ckpt or paths.GROOT_BALANCE_ONNX))
        self._walking = ort.InferenceSession(str(walking_ckpt or paths.GROOT_WALK_ONNX))
        self._standing_input = self._standing.get_inputs()[0].name
        self._walking_input = self._walking.get_inputs()[0].name

        self.default_angles_15 = DEFAULT_LOWER_QPOS.copy()
        self.default_angles_29 = np.zeros(29, dtype=np.float32)
        self.default_angles_29[:15] = self.default_angles_15

        self._last_actions: NDArray[np.float32] = np.zeros(self.N_ACTIONS, dtype=np.float32)
        self._commands: NDArray[np.float32] = np.array([0.0, 0.0, 0.0, self.HEIGHT_BASE], dtype=np.float32)
        self._waist_yrp: NDArray[np.float32] = np.zeros(3, dtype=np.float32)

        self._root_vel_filter = LPFilter(alpha=self.ROOT_VEL_FILTER_ALPHA)

        self._obs_history: deque[NDArray[np.float32]] = deque(maxlen=self.N_HISTORY)
        for _ in range(self.N_HISTORY):
            self._obs_history.append(np.zeros(self.N_OBS, dtype=np.float32))

        self._warmup_counter = 0

        # Warm up ONNX so the first real call does not pay the JIT cost.
        dummy = np.random.randn(1, self.N_OBS * self.N_HISTORY).astype(np.float32)
        for _ in range(10):
            self._standing.run(None, {self._standing_input: dummy})
            self._walking.run(None, {self._walking_input: dummy})

    def set_command(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw: float = 0.0,
        height: float = HEIGHT_BASE,
        waist_yrp: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Set the desired root-velocity, base-height and waist (yaw, roll, pitch) commands."""
        raw = np.array([vx, vy, yaw], dtype=np.float32)
        filtered = self._root_vel_filter.next(raw)
        self._commands[:3] = filtered
        self._commands[3] = height
        self._waist_yrp[:] = waist_yrp

    def reset_warmup(self) -> None:
        """Re-arm the warmup blend; call when (re)activating the policy."""
        self._warmup_counter = 0

    def _get_observation(
        self,
        dof_pos_raw: NDArray[np.float32],
        dof_vel_raw: NDArray[np.float32],
        ang_vel_raw: NDArray[np.float32],
        base_quat_wxyz: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        gravity = _quat_rotate_inverse(base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        ang_vel = ang_vel_raw * self.ANG_VEL_SCALE
        dof_pos = (dof_pos_raw - self.default_angles_29) * self.DOF_POS_SCALE
        dof_vel = dof_vel_raw * self.DOF_VEL_SCALE
        cmd_scaled = self._commands[:3] * self.CMD_SCALE

        obs = np.zeros(self.N_OBS, dtype=np.float32)
        obs[0:3] = cmd_scaled
        obs[3:4] = self._commands[3]
        obs[4:7] = self._waist_yrp
        obs[7:10] = ang_vel
        obs[10:13] = gravity
        obs[13:42] = dof_pos
        obs[42:71] = dof_vel
        obs[71:86] = self._last_actions
        return obs

    def step(
        self,
        dof_pos: NDArray[np.float32],
        dof_vel: NDArray[np.float32],
        ang_vel: NDArray[np.float32],
        base_quat_wxyz: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Run one inference; return the 15-DoF target qpos (12 legs + 3 waist)."""
        obs = self._get_observation(
            dof_pos.astype(np.float32),
            dof_vel.astype(np.float32),
            ang_vel.astype(np.float32),
            base_quat_wxyz.astype(np.float32),
        )
        self._obs_history.append(obs)
        obs_input = np.array(self._obs_history, dtype=np.float32).flatten().reshape(1, -1)

        cmd_norm = float(np.linalg.norm(self._commands[:3]))
        if cmd_norm < self.CMD_THRESHOLD:
            session, name = self._standing, self._standing_input
        else:
            session, name = self._walking, self._walking_input

        raw = session.run(None, {name: obs_input})[0].squeeze()
        clipped = np.clip(raw, -100.0, 100.0).astype(np.float32)
        scaled = clipped * self.ACTION_SCALE
        target_qpos = self.default_angles_15 + scaled[: self.N_ACTIONS]

        if self._warmup_counter < self.WARMUP_STEPS:
            ratio = self._warmup_counter / self.WARMUP_STEPS
            qdes = self.default_angles_15 * (1 - ratio) + target_qpos * ratio
            self._warmup_counter += 1
        else:
            qdes = target_qpos

        # The waist (12:15) is the commanded value, not the policy's.
        qdes[12:15] = self._waist_yrp

        self._last_actions = clipped[: self.N_ACTIONS].copy()
        result: NDArray[np.float64] = qdes.astype(np.float64)
        return result
