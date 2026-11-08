"""The 50 Hz whole-body control decision, shared by sim and real.

Both sides pelvis-compensate the wrist target, solve the upper-body IK, split it into an arm target
plus a torso command, and feed gravity comp forward. Only delivery differs: sim writes PhysX drives
in-process (:func:`deliver_to_sim`); real writes the shared-memory arrays the DDS subprocess drains.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from hug.control.config import N_HAND
from hug.utils import smooth_lerp

if TYPE_CHECKING:
    from hug.sim.backend import SapienBackend
    from hug.wbc.controller import WholeBodyController
    from hug.wbc.lower_body_policy import LowerBodyPolicy


@dataclass(frozen=True)
class BodyCommand:
    """One tick of commands. The feed-forwards are already clipped by the controller."""

    q21: NDArray[np.float32]  # [torso(4), head(3), left_arm(7), right_arm(7)]
    arm_qpos: NDArray[np.float64]  # (14,) left then right
    arm_tau_ff: NDArray[np.float64]  # (14,) gravity comp
    waist_tau_ff: NDArray[np.float64]  # (3,) gravity comp, yaw/roll/pitch; zero unless whole_body_gravity
    hand_qpos: NDArray[np.float64]  # (12,) left(6) then right(6)
    height: float  # torso height → locomotion policy
    waist_yrp: tuple[float, float, float]  # waist yaw/roll/pitch → locomotion policy


def solve_body_command(
    wbc: "WholeBodyController",
    motor_q: NDArray[np.float64],
    *,
    right_pose7: Optional[NDArray[np.float64]] = None,
    left_pose7: Optional[NDArray[np.float64]] = None,
    right_hand: Optional[NDArray[np.float64]] = None,
    left_hand: Optional[NDArray[np.float64]] = None,
) -> BodyCommand:
    """Compensate → IK → gravity comp. A ``None`` wrist is left to the IK's mask; a ``None`` hand commands zeros."""
    q21, arm_tau_ff, waist_tau_ff = wbc.step(motor_q, right_pose7, left_pose7)
    return pack_body_command(q21, arm_tau_ff, waist_tau_ff, right_hand=right_hand, left_hand=left_hand)


def pack_body_command(
    q21: NDArray[np.float32],
    arm_tau_ff: NDArray[np.float64],
    waist_tau_ff: NDArray[np.float64],
    *,
    right_hand: Optional[NDArray[np.float64]] = None,
    left_hand: Optional[NDArray[np.float64]] = None,
) -> BodyCommand:
    """Split a solved ``(q21, arm_tau_ff, waist_tau_ff)`` into the fields delivery needs."""
    hand_qpos = np.zeros(N_HAND, dtype=np.float64)
    if left_hand is not None:
        hand_qpos[:6] = left_hand
    if right_hand is not None:
        hand_qpos[6:] = right_hand
    return BodyCommand(
        q21=q21,
        arm_qpos=q21[7:21].astype(np.float64),
        arm_tau_ff=np.asarray(arm_tau_ff, dtype=np.float64),
        waist_tau_ff=np.asarray(waist_tau_ff, dtype=np.float64),
        hand_qpos=hand_qpos,
        height=float(q21[0]),
        waist_yrp=(float(q21[1]), float(q21[2]), float(q21[3])),
    )


def body_command_from_q21(
    wbc: "WholeBodyController",
    q21: NDArray[np.float32],
    *,
    right_hand: Optional[NDArray[np.float64]] = None,
    left_hand: Optional[NDArray[np.float64]] = None,
) -> BodyCommand:
    """Pack an explicit posture, bypassing the IK (glides and holds), with the same gravity comp."""
    return pack_body_command(q21, *wbc.gravity_comp(q21), right_hand=right_hand, left_hand=left_hand)


def glide_q21_commands(
    wbc: "WholeBodyController",
    q21_from: NDArray[np.float64],
    q21_to: NDArray[np.float64],
    n_ticks: int,
    *,
    right_hand: Optional[NDArray[np.float64]] = None,
    left_hand: Optional[NDArray[np.float64]] = None,
    interp_order: int = 5,
) -> Iterator[BodyCommand]:
    """Yield ``n_ticks`` joint-space commands from ``q21_from`` to ``q21_to`` (reached on the last), hands held.

    EE space cannot undo a grasp (redundant IK); the caller ticks the compensator, then ``wbc.reset(q21_to)``."""
    q_from = np.asarray(q21_from, dtype=np.float64)
    q_to = np.asarray(q21_to, dtype=np.float64)
    n = max(int(n_ticks), 1)
    for k in range(n):
        q21 = smooth_lerp(q_from, q_to, (k + 1) / n, order=interp_order)
        yield body_command_from_q21(wbc, q21.astype(np.float32), right_hand=right_hand, left_hand=left_hand)


def deliver_to_sim(
    backend: "SapienBackend",
    lower: "LowerBodyPolicy",
    cmd: BodyCommand,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    yaw: float = 0.0,
) -> None:
    """Sim delivery: run the locomotion policy, write the drives, step physics."""
    s = backend.read_state()
    lower.set_command(vx=vx, vy=vy, yaw=yaw, height=cmd.height, waist_yrp=cmd.waist_yrp)
    qdes15 = lower.step(s.motor_q, s.motor_qd, s.gyro, s.base_quat)
    backend.set_command(qdes15, cmd.arm_qpos, cmd.arm_tau_ff, cmd.hand_qpos, cmd.waist_tau_ff)
    backend.step()
