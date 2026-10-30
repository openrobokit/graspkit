"""Arm/hand/camera factories for `RealEnv` by config name; imports retargetingkit, so needs the `xarm` extra."""

import time

import numpy as np
from retargetingkit.targets.arms.xarm6 import XArm6, XArm6Config
from retargetingkit.targets.arms.xarm7 import XArm7, XArm7Config
from retargetingkit.targets.hands.ability_hand import AbilityHand, AbilityHandConfig
from retargetingkit.targets.hands.sharpa_hand import SharpaHand
from retargetingkit.targets.hands.xhand import XHand, XHandConfig

from graspkit.configs.real_env_config import RealEnvConfig


def _started(device):
    device.start()
    return device


class _WaitXArm6(XArm6):
    """xArm set_qpos with RealEnv's `wait=` contract (velocity-capped; poll until reached)."""

    def set_qpos(self, q: np.ndarray, wait: bool = False):
        super().set_qpos(q)
        while wait and np.max(np.abs(self.get_qpos() - q)) > 0.01:
            time.sleep(0.1)

    def set_qpos_traj(self, traj: np.ndarray, hz: float):
        for q in traj:
            super().set_qpos(q)
            time.sleep(1 / hz)


class _WaitXArm7(XArm7):
    """xArm set_qpos with RealEnv's `wait=` contract (velocity-capped; poll until reached)."""

    def set_qpos(self, q: np.ndarray, wait: bool = False):
        super().set_qpos(q)
        while wait and np.max(np.abs(self.get_qpos() - q)) > 0.01:
            time.sleep(0.1)

    def set_qpos_traj(self, traj: np.ndarray, hz: float):
        for q in traj:
            super().set_qpos(q)
            time.sleep(1 / hz)


def _make_realsense():
    from retargetingkit.sources.cameras.realsense import Realsense, RealsenseConfig

    return _started(Realsense(RealsenseConfig()))


xarm7_xhand_right = RealEnvConfig(
    name="xarm7_xhand_right",
    robot_config_name="xarm7_xhand_right",
    make_arm=lambda: _started(
        _WaitXArm7(XArm7Config(ip="192.168.1.242", max_velocity=np.array([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 1.5])))
    ),
    make_hand=lambda: _started(
        XHand(XHandConfig(serial_port="/dev/serial/by-id/usb-FTDI_USB-RS485-WE_FTRJCBQC-if00-port0"))
    ),
    workspace_center=(0.41, 0.37, -0.02),
    make_camera=_make_realsense,
)

xarm6_sharpa_right = RealEnvConfig(
    name="xarm6_sharpa_right",
    robot_config_name="xarm6_sharpa_right",
    make_arm=lambda: _started(
        _WaitXArm6(XArm6Config(ip="192.168.1.242", max_velocity=np.array([0.4, 0.4, 0.4, 0.5, 0.5, 1.5])))
    ),
    # SharpaHand connects inside __init__ and has no .start(), so it is NOT wrapped in _started.
    make_hand=lambda: SharpaHand(speed_coeff=0.3, current_coeff=0.3, device_sn=None),
    workspace_center=(0.41, 0.37, -0.02),
    make_camera=_make_realsense,
)

xarm7_ability_right = RealEnvConfig(
    name="xarm7_ability_right",
    robot_config_name="xarm7_ability_right",
    make_arm=lambda: _started(
        _WaitXArm7(XArm7Config(ip="192.168.1.242", max_velocity=np.array([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 1.5])))
    ),
    make_hand=lambda: _started(
        AbilityHand(
            AbilityHandConfig(
                port="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BUBZb11A922-if00-port0"
            )
        )
    ),
    workspace_center=(0.41, 0.37, -0.02),
    make_camera=_make_realsense,
    hand_reindex=[2, 3, 4, 5, 1, 0],  # ours (robot_helper order) -> Ability hardware channel order
)


# Explicit dict (NOT lazy_find_instances): a missing `xarm` extra must raise here, not be swallowed.
REAL_ENV_CONFIGS = {c.name: c for c in (xarm7_xhand_right, xarm6_sharpa_right, xarm7_ability_right)}
