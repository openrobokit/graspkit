"""Real-side delivery and the two blocking ramps at the edges of a session.

The interface subprocess owns the DDS write and its watchdog: the main loop must call
step_to_target / deliver_to_real every tick or the arms go limp. Only the bring-up ramp
and the shutdown block here, before the loop exists and after it has exited.
"""

from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin

from hug.control.config import HZ, IDLE_HAND, N_HAND, q21_posture
from hug.control.step import BodyCommand, solve_body_command
from hug.utils import Rate, smooth_lerp

if TYPE_CHECKING:
    from hug.real.g1_inspire_interface import G1InspireInterface
    from hug.wbc.controller import WholeBodyController


def tick_compensator(wbc: "WholeBodyController", motor_q: np.ndarray) -> None:
    """Tick the compensator's rate limiter while the IK is out of the loop, so its next real call sees no step."""
    wbc.compensator.correct(motor_q, pin.SE3.Identity())


def step_to_target(
    interface: "G1InspireInterface",
    wbc: "WholeBodyController",
    right_pose7: np.ndarray,
    hand_6: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One tick: the shared decision for the right hand, delivered to the interface. Returns ``(q21, tau_ff)``."""
    cmd = solve_body_command(
        wbc,
        interface.get_motor_qpos(),
        right_pose7=right_pose7,
        right_hand=hand_6,
    )
    deliver_to_real(interface, cmd)
    return cmd.q21, cmd.arm_tau_ff


def deliver_to_real(interface: "G1InspireInterface", cmd: BodyCommand) -> None:
    """Write one command to the shared-memory arrays the subprocess drains onto DDS; kp stays at the default."""
    interface.set_arm_target_qpos(cmd.arm_qpos)
    interface.set_locomotion_command(height=cmd.height, waist_yrp=cmd.waist_yrp, waist_tau=tuple(cmd.waist_tau_ff))
    interface.set_hand_target_qpos(cmd.hand_qpos)
    interface.set_arm_kp_kd_tau(tau=cmd.arm_tau_ff)


def graceful_shutdown(interface: "G1InspireInterface", wbc: "WholeBodyController", ramp_s: float = 3.0) -> None:
    """Relax the arms over ``ramp_s`` (kp and the waist feed-forward to zero), the torso held as last commanded.

    Then the subprocess ends and nothing balances the robot: the operator has its weight on the rope by now."""
    print(f"[shutdown] relaxing the arms over {ramp_s:.1f} s, torso held ...")
    default_kp, _, _ = interface.get_arm_kp_kd_tau()
    default_kp = np.asarray(default_kp, dtype=np.float64).copy()
    arm_q_cmd = interface.get_motor_qpos()[15:29]
    loco = interface.get_locomotion_command()
    height, waist_yrp = float(loco[3]), (float(loco[4]), float(loco[5]), float(loco[6]))
    hand_cmd = np.zeros(N_HAND, dtype=np.float64)
    hand_cmd[6:] = IDLE_HAND
    n_steps = max(int(ramp_s * HZ), 1)
    rate = Rate(HZ)
    for k in range(n_steps):
        alpha = (k + 1) / n_steps
        kp = (1.0 - alpha) * default_kp
        q21 = q21_posture(height, arm_q_cmd[:7], arm_q_cmd[7:])
        q21[1:4] = waist_yrp
        arm_tau, waist_tau = wbc.gravity_comp(q21)
        interface.set_arm_target_qpos(arm_q_cmd)
        # the waist feed-forward (zero unless whole_body_gravity) fades out with the arm kp, not in one step
        interface.set_locomotion_command(height=height, waist_yrp=waist_yrp, waist_tau=tuple((1.0 - alpha) * waist_tau))
        interface.set_hand_target_qpos(hand_cmd)
        interface.set_arm_kp_kd_tau(kp=kp, tau=arm_tau)
        rate.sleep()
    interface.cleanup()


def smooth_init_to_ready(
    interface: "G1InspireInterface",
    wbc: "WholeBodyController",
    q21_target: np.ndarray,
    duration_s: float = 3.0,
    interp_order: int = 5,
) -> None:
    """Quintic ramp of the arms from wherever they are into ``q21_target`` at full kp, then seed the IK there."""
    q21 = np.asarray(q21_target, dtype=np.float32)
    height, target = float(q21[0]), q21[7:21].astype(np.float64)
    current = interface.get_motor_qpos()[15:29]
    hand_cmd = np.zeros(N_HAND, dtype=np.float64)
    hand_cmd[6:] = IDLE_HAND
    n_steps = max(int(duration_s * HZ), 1)
    rate = Rate(HZ)
    print(f"[init] arms → READY over {duration_s:.1f} s ...")
    for k in range(n_steps):
        alpha = (k + 1) / n_steps
        arm_q = smooth_lerp(current, target, alpha, order=interp_order)
        arm_tau, waist_tau = wbc.gravity_comp(q21_posture(height, arm_q[:7], arm_q[7:]))
        interface.set_arm_target_qpos(arm_q)
        # the waist feed-forward (zero unless whole_body_gravity) fades in, so the first tracking tick sees no step
        interface.set_locomotion_command(height=height, waist_yrp=(0.0, 0.0, 0.0), waist_tau=tuple(alpha * waist_tau))
        interface.set_hand_target_qpos(hand_cmd)
        interface.set_arm_kp_kd_tau(tau=arm_tau)
        tick_compensator(wbc, interface.get_motor_qpos())
        rate.sleep()
    wbc.ik.reset(q21)
