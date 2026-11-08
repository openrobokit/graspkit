"""Blocking move-to-qpos helper for the G1 arms, used by upper-body calibration.

Ramps the 14 arm DoFs (motors 15..28) at a bounded velocity under a stiff PD hold; legs and waist are
damping only (kp=0). Use only with the robot suspended: no locomotion, no balance loop.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from hug.control.config import ARM_KD, ARM_KP
from hug.real.g1_robot import G1_NUM_MOTOR, G1JointIndex, G1Robot

ARM_JOINT_INDICES: NDArray[np.intp] = np.arange(
    G1JointIndex.LeftShoulderPitch, G1JointIndex.RightWristYaw + 1, dtype=np.intp
)


@dataclass
class G1ArmMoverConfig:
    # Per-joint or scalar; both broadcast onto the 14 arm DoFs. Default: the stack's arm PD (control.config).
    arm_kp: Union[float, NDArray[np.float64]] = field(default_factory=lambda: ARM_KP.copy())
    arm_kd: Union[float, NDArray[np.float64]] = field(default_factory=lambda: ARM_KD.copy())
    nonarm_kd: float = 2.0
    settle_q_tol: float = 0.035
    settle_dq_tol: float = 0.05
    settle_timeout_s: float = 8.0
    settle_hold_s: float = 0.3
    interp_velocity: float = 0.5
    min_duration_s: float = 0.5


class G1ArmMover:
    """Blocking move-to-qpos wrapper for G1's 14 arm DoFs.

    Call ``hold_current()`` before the first move so PD latches the current angles instead of zero."""

    def __init__(self, robot: G1Robot, config: Optional[G1ArmMoverConfig] = None) -> None:
        self.robot = robot
        self.config = config or G1ArmMoverConfig()
        self._kp: NDArray[np.float64] = np.zeros(G1_NUM_MOTOR)
        self._kp[ARM_JOINT_INDICES] = self.config.arm_kp
        self._kd: NDArray[np.float64] = np.full(G1_NUM_MOTOR, self.config.nonarm_kd)
        self._kd[ARM_JOINT_INDICES] = self.config.arm_kd
        self._zeros: NDArray[np.float64] = np.zeros(G1_NUM_MOTOR)
        self._latched_q_des: NDArray[np.float64] = np.zeros(G1_NUM_MOTOR)

    def hold_current(self) -> None:
        """Latch q_des to the current joint angles and apply arm PD + leg damping."""
        self._latched_q_des = self.robot.GetMotorQ()
        self.robot.SetJointCmd(
            q=self._latched_q_des,
            dq=self._zeros,
            tau=self._zeros,
            kp=self._kp,
            kd=self._kd,
        )
        self.robot.Step()

    def move_to_arm_qpos(
        self,
        target_arm_q: NDArray[np.float64],
        max_velocity: Optional[float] = None,
        settle: bool = True,
    ) -> None:
        """Ramp the arms to ``target_arm_q`` (14,) at ``max_velocity`` rad/s, then PD-hold.

        With ``settle``, holds until within tolerance for ``settle_hold_s`` or raises after ``settle_timeout_s``."""
        target = np.asarray(target_arm_q, dtype=np.float64)
        assert target.shape == (ARM_JOINT_INDICES.size,), (
            f"target_arm_q must be shape ({ARM_JOINT_INDICES.size},), got {target.shape}"
        )
        v = float(max_velocity if max_velocity is not None else self.config.interp_velocity)

        current_full = self.robot.GetMotorQ()
        current_arm = current_full[ARM_JOINT_INDICES]
        delta = target - current_arm
        max_delta = float(np.max(np.abs(delta)))
        duration = max(max_delta / v, self.config.min_duration_s)
        n_steps = max(int(duration / self.robot._control_dt), 2)

        q_des_full = current_full.copy()
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            q_des_full[ARM_JOINT_INDICES] = current_arm + alpha * delta
            self.robot.SetJointCmd(q=q_des_full, dq=self._zeros, tau=self._zeros, kp=self._kp, kd=self._kd)
            self.robot.Step()

        self._latched_q_des = q_des_full
        if not settle:
            return

        t0 = time.time()
        within_since: Optional[float] = None
        while True:
            self.robot.SetJointCmd(q=self._latched_q_des, dq=self._zeros, tau=self._zeros, kp=self._kp, kd=self._kd)
            self.robot.Step()
            q_now = self.robot.GetMotorQ()[ARM_JOINT_INDICES]
            dq_now = self.robot.GetMotorVel()[ARM_JOINT_INDICES]
            err = float(np.max(np.abs(q_now - target)))
            vel = float(np.max(np.abs(dq_now)))
            in_tol = err < self.config.settle_q_tol and vel < self.config.settle_dq_tol
            now = time.time()
            if in_tol:
                if within_since is None:
                    within_since = now
                elif now - within_since > self.config.settle_hold_s:
                    return
            else:
                within_since = None
            if now - t0 > self.config.settle_timeout_s:
                raise RuntimeError(
                    f"G1 arms did not settle within {self.config.settle_timeout_s}s; "
                    f"final err={err:.4f} rad, vel={vel:.4f} rad/s"
                )

    def get_arm_qpos(self) -> NDArray[np.float64]:
        return self.robot.GetMotorQ()[ARM_JOINT_INDICES].copy()

    def hold(self, duration_s: float) -> None:
        """Tick PD at control rate for ``duration_s``, holding the latched target.

        Use this, not ``time.sleep``, to wait between motions: the G1 watchdog (~100-200 ms) drops the motors."""
        n_steps = max(int(duration_s / self.robot._control_dt), 1)
        for _ in range(n_steps):
            self.robot.SetJointCmd(
                q=self._latched_q_des,
                dq=self._zeros,
                tau=self._zeros,
                kp=self._kp,
                kd=self._kd,
            )
            self.robot.Step()
