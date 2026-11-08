"""Teleop tuning: the operator's workspace, target slew rates and grip endpoints."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hug.sim.scene import STAND_HAND_QPOS
from hug.utils import slew_pose7

HAND_CLOSED = np.array([0.6, 0.8, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)


def clamp_workspace(
    pose7: NDArray[np.float64],
    x: tuple[float, float],
    y: tuple[float, float],
    z: tuple[float, float],
) -> NDArray[np.float64]:
    """Clamp a target's translation into a box."""
    out = pose7.copy()
    for i, (lo, hi) in enumerate((x, y, z)):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


@dataclass
class TeleopConfig:
    """What the operator can reach, how fast targets may move, and the grip endpoints."""

    # Workspace box for dragged targets, in the controller frame. Generous: the IK's joint limits are the real bound.
    ws_x: tuple[float, float] = (0.10, 0.70)
    ws_y: tuple[float, float] = (-0.60, 0.60)
    ws_z: tuple[float, float] = (0.30, 1.35)

    slew_lin: float = 0.40  # m/s, so a fast drag reaches the IK as a continuous motion
    slew_ang: float = 1.50  # rad/s

    hand_open: NDArray[np.float64] = field(default_factory=lambda: STAND_HAND_QPOS.copy())
    hand_closed: NDArray[np.float64] = field(default_factory=lambda: HAND_CLOSED.copy())

    def grip(self, amount: float) -> NDArray[np.float64]:
        """6 proximal joints, interpolated from open (0.0) to closed (1.0)."""
        return self.hand_open * (1.0 - amount) + self.hand_closed * amount

    def track(
        self, cmd_pose7: NDArray[np.float64], target_pose7: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        """Step the commanded target toward ``target_pose7``: clamp into the workspace, then slew."""
        target = clamp_workspace(target_pose7, self.ws_x, self.ws_y, self.ws_z)
        return slew_pose7(cmd_pose7, target, dt, self.slew_lin, self.slew_ang)
