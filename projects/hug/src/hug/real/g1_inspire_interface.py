"""G1 + dual Inspire-hand interface: a 50 Hz subprocess that owns the DDS write.

Each tick it drains the shared-memory arm / hand / locomotion arrays onto the robot (arms under a
per-tick velocity limit), optionally runs the GR00T lower-body policy, mirrors motor qpos and the remote
back, and stops on the remote's X button. Set the multiprocessing start method to ``spawn`` first.
"""

import multiprocessing as mp
import signal
import time
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from hug.control.config import ARM_KD, ARM_KP, RealControlConfig
from hug.real.g1_robot import G1Robot, RemoteState, channel_factory_init
from hug.real.inspire_hand import InspireHand
from hug.utils import blend_waypoints
from hug.wbc.lower_body_policy import (
    DEFAULT_LOWER_KD,
    DEFAULT_LOWER_KP,
    DEFAULT_LOWER_QPOS,
    LowerBodyPolicy,
)

LOWER_INDICES = list(range(0, 15))  # 12 legs + 3 waist
LEFT_ARM_INDICES = list(range(15, 22))
RIGHT_ARM_INDICES = list(range(22, 29))
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
N_LOWER = 15
N_ARM = 14
N_MOTOR = N_LOWER + N_ARM
N_FINGER = 12  # 6 per hand × 2
N_LOCO_CMD = 10  # vx, vy, yaw, height, waist yaw/roll/pitch, waist feed-forward yaw/roll/pitch (N·m)

# RemoteState field order: four floats, then ints.
_REMOTE_FIELDS = (
    "lx",
    "ly",
    "rx",
    "ry",
    "R1",
    "L1",
    "Start",
    "Select",
    "R2",
    "L2",
    "F1",
    "F3",
    "A",
    "B",
    "X",
    "Y",
    "Up",
    "Right",
    "Down",
    "Left",
)
N_REMOTE = len(_REMOTE_FIELDS)

# Lower-body policy runs at 50 Hz; the subprocess control loop matches.
UB_FREQ = 50

# fmt: off
_DEFAULT_ARMS_QPOS: NDArray[np.float64] = np.array([
    0.0, 0.0, 0.0, 0.0,  1.57, 0.0, 0.0,   # left arm
    0.0, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0,   # right arm
])
# _start_arm's waypoint: elbows folded back so the arms clear the body on the way to default_arms_qpos
_ARM_HIGH_QPOS: NDArray[np.float64] = np.array([
    0.0,  0.3,  1.5, -0.8, 0.0, 0.0, 0.0,  # left arm raised
    0.0, -0.3, -1.5, -0.8, 0.0, 0.0, 0.0,  # right arm raised
])
# fmt: on


@dataclass
class G1InspireConfig:
    network_interface: str = ""  # DDS interface; the subprocess opens its own participant
    upper_dry_run: bool = False  # skip the arm bring-up and never write arm commands
    hand_dry_run: bool = False  # skip both hands
    enable_walking_policy: bool = False  # run the GR00T lower-body policy in the subprocess
    default_arms_qpos: NDArray[np.float64] = field(default_factory=lambda: _DEFAULT_ARMS_QPOS.copy())  # after bring-up
    left_hand_serial_port: str = ""  # empty = no left hand
    right_hand_serial_port: str = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"  # empty = no right hand
    arm_vel_limit_rad_per_step: Optional[float] = RealControlConfig.arm_vel_limit_rad_per_step  # max step per tick

    def make_robot(self, read_only: bool = False) -> G1Robot:
        """The DDS robot backend: domain 0, topic prefix ``rt``."""
        channel_factory_init(self.network_interface, domain_id=0)
        return G1Robot(read_only=read_only, topic_prefix="rt")

    def make_hands(self) -> tuple[Optional[InspireHand], Optional[InspireHand]]:
        """Construct left and right hand drivers. Returns None for unspecified ports."""
        left = InspireHand(serial_port=self.left_hand_serial_port, id=1) if self.left_hand_serial_port else None
        right = InspireHand(serial_port=self.right_hand_serial_port, id=1) if self.right_hand_serial_port else None
        return left, right


class G1InspireInterface:
    # The arm PD the stack is tuned around; sim mirrors it. One definition, in control.config.
    default_arms_kp: NDArray[np.float64] = ARM_KP
    default_arms_kd: NDArray[np.float64] = ARM_KD

    def __init__(self, config: G1InspireConfig) -> None:
        self.config = config
        self.default_arms_qpos: NDArray[np.float64] = config.default_arms_qpos.copy()

        self.shared_arm_target_qpos: Any = mp.Array("d", N_ARM)
        self.shared_arm_target_qpos[:] = self.default_arms_qpos
        self.shared_arm_kp: Any = mp.Array("d", list(self.default_arms_kp))
        self.shared_arm_kd: Any = mp.Array("d", list(self.default_arms_kd))
        self.shared_arm_tau: Any = mp.Array("d", N_ARM)

        # layout as N_LOCO_CMD; height starts at 0.74 m, standing
        self.shared_loco_cmd: Any = mp.Array("d", N_LOCO_CMD)
        self.shared_loco_cmd[:] = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.shared_finger_cmd: Any = mp.Array("d", N_FINGER)
        self.shared_finger_qpos: Any = mp.Array("d", N_FINGER)

        self.shared_motor_qpos: Any = mp.Array("d", N_MOTOR)
        self.shared_remote_state: Any = mp.Array("d", N_REMOTE)

        self.arm_ready: MpEvent = mp.Event()
        self.should_stop: MpEvent = mp.Event()

        # Subprocess attrs (only valid inside _exec_loop)
        self.robot: Optional[G1Robot] = None
        self.left_hand: Optional[InspireHand] = None
        self.right_hand: Optional[InspireHand] = None
        self.policy: Optional[LowerBodyPolicy] = None

        self.process = mp.Process(
            target=self._exec_loop,
            args=(self.arm_ready, self.should_stop),
            daemon=True,
        )
        self.process.start()

        self.arm_ready.wait()
        print("[g1] interface ready.")

    # ============================ subprocess code ============================
    def _exec_loop(self, arm_ready: MpEvent, should_stop: MpEvent) -> None:
        # A terminal Ctrl+C reaches this process too; it belongs to the main one, which ramps down, then ends this one
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.arm_ready = arm_ready
        self.should_stop = should_stop

        self.robot = self.config.make_robot(read_only=False)

        if not self.config.hand_dry_run:
            self.left_hand, self.right_hand = self.config.make_hands()

        # Before the first command: a load (or first-use download) between the ramps would leave the legs unfed
        if self.config.enable_walking_policy:
            self.policy = LowerBodyPolicy()

        print("[g1] standing up ...")
        self._stand(wait=False)

        if not self.config.upper_dry_run:
            print("[g1] raising the arms ...")
            self._start_arm()

        with self.shared_motor_qpos.get_lock():
            self.shared_motor_qpos[:] = self.robot.GetMotorQ()

        self.arm_ready.set()

        try:
            while not self.should_stop.is_set():
                t0 = time.time()
                if not self._step():
                    break
                dt = time.time() - t0
                if dt < 1.0 / UB_FREQ:
                    time.sleep(1.0 / UB_FREQ - dt)
        finally:
            if self.left_hand is not None:
                self.left_hand.close()
                print("[hand] left disconnected.")
            if self.right_hand is not None:
                self.right_hand.close()
                print("[hand] right disconnected.")

    def _step(self) -> bool:
        assert self.robot is not None

        motor_q = self.robot.GetMotorQ()
        with self.shared_motor_qpos.get_lock():
            self.shared_motor_qpos[:] = motor_q

        remote = self.robot.GetRemoteState()
        with self.shared_remote_state.get_lock():
            self.shared_remote_state[:] = [getattr(remote, f) for f in _REMOTE_FIELDS]

        with self.shared_arm_target_qpos.get_lock():
            arm_target = np.copy(self.shared_arm_target_qpos[:])

        finger_cmd = np.zeros(N_FINGER)
        if not self.config.hand_dry_run:
            with self.shared_finger_cmd.get_lock():
                finger_cmd = np.copy(self.shared_finger_cmd[:])
            with self.shared_finger_qpos.get_lock():
                if self.left_hand is not None:
                    self.shared_finger_qpos[:6] = self.left_hand.GetPosition()
                if self.right_hand is not None:
                    self.shared_finger_qpos[6:] = self.right_hand.GetPosition()

        # Safety stop on X
        if remote.X:
            return False

        if not self.config.hand_dry_run:
            if self.left_hand is not None:
                self.left_hand.SetPosition(finger_cmd[:6])
            if self.right_hand is not None:
                self.right_hand.SetPosition(finger_cmd[6:])

        if not self.config.upper_dry_run:
            lim, arm_q = self.config.arm_vel_limit_rad_per_step, motor_q[ARM_INDICES]
            arm_limited = arm_target if lim is None else arm_q + np.clip(arm_target - arm_q, -lim, lim)
            self.robot.SetPartialJointCmd(
                np.array(ARM_INDICES),
                arm_limited.copy(),
                np.zeros(N_ARM),
                np.array(self.shared_arm_tau[:]),
                np.array(self.shared_arm_kp[:]),
                np.array(self.shared_arm_kd[:]),
            )

        if self.policy is not None:
            with self.shared_loco_cmd.get_lock():
                cmd = np.array(self.shared_loco_cmd[:])
            self.policy.set_command(
                vx=float(cmd[0]),
                vy=float(cmd[1]),
                yaw=float(cmd[2]),
                height=float(cmd[3]),
                waist_yrp=(float(cmd[4]), float(cmd[5]), float(cmd[6])),
            )
            qd = self.robot.GetMotorVel().astype(np.float32)
            ang_vel = self.robot.GetAngVel().astype(np.float32)
            base_quat = self.robot.GetBaseQuat().astype(np.float32)
            qdes = self.policy.step(motor_q.astype(np.float32), qd, ang_vel, base_quat)
            tau = np.zeros(N_LOWER)
            tau[12:15] = cmd[7:10]  # waist gravity feed-forward; zero unless whole_body_gravity
            self.robot.SetPartialJointCmd(
                np.array(LOWER_INDICES),
                qdes,
                np.zeros(N_LOWER),
                tau,
                DEFAULT_LOWER_KP,
                DEFAULT_LOWER_KD,
            )

        self.robot.Step(do_sleep=False)
        return True

    def _start_arm(self, steps: int = 400) -> None:
        """Ramp the arms current → ``_ARM_HIGH_QPOS`` → ``default_arms_qpos`` at the default arm kp/kd.

        No gravity compensation and no per-tick velocity limit: the ramp itself is trusted to be smooth."""
        assert self.robot is not None
        current = self.robot.GetMotorQ()[ARM_INDICES]
        waypoints = [current, _ARM_HIGH_QPOS, self.default_arms_qpos]
        for t in range(steps):
            ratio = t / max(steps - 1, 1)
            target = blend_waypoints(waypoints, ratio)
            self.robot.SetPartialJointCmd(
                np.array(ARM_INDICES),
                target,
                np.zeros(N_ARM),
                np.zeros(N_ARM),
                self.default_arms_kp,
                self.default_arms_kd,
            )
            self.robot.Step(do_sleep=False)
            time.sleep(0.01)

    def _stand(self, steps: int = 200, ramp_steps: int = 150, wait: bool = True) -> None:
        """Ramp the lower body current → DEFAULT_LOWER_QPOS at the walking-policy gains.

        With ``wait``, blocks after the ramp until the operator presses L1 on the remote."""
        assert self.robot is not None
        target = DEFAULT_LOWER_QPOS.astype(np.float64)
        for t in range(steps):
            current = self.robot.GetMotorQ()[LOWER_INDICES]
            ratio = min(t / ramp_steps, 1.0)
            action = (1 - ratio) * current + ratio * target
            self.robot.SetPartialJointCmd(
                np.array(LOWER_INDICES),
                action,
                np.zeros(N_LOWER),
                np.zeros(N_LOWER),
                DEFAULT_LOWER_KP,
                DEFAULT_LOWER_KD,
            )
            self.robot.Step(do_sleep=False)
            time.sleep(0.01)

        if wait:
            print("[g1] in stand pose — press L1 on the remote to continue.")
            while not self.should_stop.is_set():
                t0 = time.time()
                if self.robot.GetRemoteState().L1 == 1:
                    print("[g1] L1 received.")
                    break
                dt = time.time() - t0
                if dt < 1 / 50:
                    time.sleep(1 / 50 - dt)

    # ============================== main process ============================
    def set_arm_target_qpos(self, qpos: NDArray[np.float64]) -> None:
        with self.shared_arm_target_qpos.get_lock():
            self.shared_arm_target_qpos[:] = qpos

    def get_motor_qpos(self) -> NDArray[np.float64]:
        with self.shared_motor_qpos.get_lock():
            return np.array(self.shared_motor_qpos[:])

    def get_arm_qpos(self) -> NDArray[np.float64]:
        return self.get_motor_qpos()[ARM_INDICES]

    def set_hand_target_qpos(self, qpos: NDArray[np.float64]) -> None:
        with self.shared_finger_cmd.get_lock():
            self.shared_finger_cmd[:] = qpos

    def get_hand_qpos(self) -> NDArray[np.float64]:
        with self.shared_finger_qpos.get_lock():
            return np.array(self.shared_finger_qpos[:])

    def get_remote_state(self) -> RemoteState:
        with self.shared_remote_state.get_lock():
            vals = list(self.shared_remote_state[:])
        return RemoteState(**{f: vals[i] if i < 4 else int(vals[i]) for i, f in enumerate(_REMOTE_FIELDS)})

    def set_arm_kp_kd_tau(
        self,
        kp: Optional[NDArray[np.float64]] = None,
        kd: Optional[NDArray[np.float64]] = None,
        tau: Optional[NDArray[np.float64]] = None,
    ) -> None:
        with self.shared_arm_kp.get_lock():
            if kp is not None:
                assert len(kp) == N_ARM
                self.shared_arm_kp[:] = kp
            if kd is not None:
                assert len(kd) == N_ARM
                self.shared_arm_kd[:] = kd
            if tau is not None:
                assert len(tau) == N_ARM
                self.shared_arm_tau[:] = tau

    def get_arm_kp_kd_tau(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        with self.shared_arm_kp.get_lock():
            return (
                np.array(self.shared_arm_kp[:]),
                np.array(self.shared_arm_kd[:]),
                np.array(self.shared_arm_tau[:]),
            )

    def set_locomotion_command(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw: float = 0.0,
        height: float = 0.74,
        waist_yrp: tuple[float, float, float] = (0.0, 0.0, 0.0),
        waist_tau: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Push a locomotion command to the subprocess; a no-op unless ``enable_walking_policy`` was set."""
        with self.shared_loco_cmd.get_lock():
            self.shared_loco_cmd[0] = vx
            self.shared_loco_cmd[1] = vy
            self.shared_loco_cmd[2] = yaw
            self.shared_loco_cmd[3] = height
            self.shared_loco_cmd[4] = waist_yrp[0]
            self.shared_loco_cmd[5] = waist_yrp[1]
            self.shared_loco_cmd[6] = waist_yrp[2]
            self.shared_loco_cmd[7:10] = list(waist_tau)

    def get_locomotion_command(self) -> NDArray[np.float64]:
        """The last locomotion command, in ``N_LOCO_CMD`` layout."""
        with self.shared_loco_cmd.get_lock():
            return np.array(self.shared_loco_cmd[:])

    def cleanup(self) -> None:
        if hasattr(self, "process") and self.process.is_alive():
            self.process.terminate()
            self.process.join()
