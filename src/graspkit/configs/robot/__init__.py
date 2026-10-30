"""RobotConfig registry: name -> module basename (explicit; no discovery, no import-time fetch)."""

import importlib

from graspkit.configs.robot_config import RobotConfig

_ROBOT_MODULES: dict[str, str] = {
    "ability_right": "ability",
    "agilex_aloha_arm": "agilex_aloha",
    "agilex_aloha_gripper": "agilex_aloha",
    "agilex_aloha_arm_mj": "agilex_aloha_mj",
    "agilex_aloha_gripper_mj": "agilex_aloha_mj",
    "allegro_right": "allegro",
    "arx_x5_arm": "arx_x5",
    "arx_x5_gripper": "arx_x5",
    "arx_x5_arm_mj": "arx_x5_mj",
    "arx_x5_gripper_mj": "arx_x5_mj",
    "arx_x5_gripper_robodojo": "arx_x5_gripper_robodojo",
    "barrett_right": "barrett",
    "dclaw_right": "dclaw",
    "franka_gripper": "franka_gripper",
    "g1_inspire_right": "g1_inspire",
    "g1_inspire_right_side_init": "g1_inspire",
    "g1_inspire_right_wbc": "g1_inspire_wbc",
    "g1_inspire_right_wbc_side_init": "g1_inspire_wbc",
    "g1_mano_right": "g1_mano",
    "inspire_right": "inspire",
    "leap_right": "leap",
    "mano_right": "mano",
    "orca_right": "orca",
    "paxini_right": "paxini",
    "panda_allegro_right_mj": "panda_allegro_mj",
    "panda_allegro_hand_mj": "panda_allegro_mj",
    "yam_gripper": "yam_gripper",
    "robotiq_2f85": "robotiq_2f85",
    "schunk_right": "schunk",
    "shadow_right": "shadow",
    "shadow_right_mj": "shadow_mj",
    "sharpa_right": "sharpa",
    "sharpa_left": "sharpa",
    "vega_sharpa_right": "vega_sharpa",
    "vega_sharpa_left": "vega_sharpa",
    "vega_sharpa_right_mj": "vega_sharpa_mj",
    "vega_sharpa_left_mj": "vega_sharpa_mj",
    "xarm6_sharpa_right": "xarm6_sharpa",
    "xarm7_ability_right": "xarm7_ability",
    "xarm7_ability_right_side_init": "xarm7_ability",
    "xarm7_ability_right_centered": "xarm7_ability",
    "xarm7_mano_right": "xarm7_mano",
    "xarm7_xhand_right": "xarm7_xhand",
    "xarm7_xhand_right_centered": "xarm7_xhand",
    "xarm7_xhand_right_mj": "xarm7_xhand_mj",
    "xarm7_shadow_right": "xarm7_shadow",
    "xarm7_sharpa_right": "xarm7_sharpa",
    "xhand_right": "xhand",
    "xhand_right_mj": "xhand_mj",
}


def build_robot_config(name: str) -> RobotConfig:
    if name not in _ROBOT_MODULES:
        raise KeyError(f"unknown robot config '{name}'; known: {sorted(_ROBOT_MODULES)}")
    return getattr(importlib.import_module(f"graspkit.configs.robot.{_ROBOT_MODULES[name]}"), name)


def has_robot_config(name: str) -> bool:
    """Whether a hand-written module defines `name`; arm+hand rigs the gallery assembles itself do not."""
    return name in _ROBOT_MODULES


def all_robot_configs() -> dict[str, RobotConfig]:
    return {name: build_robot_config(name) for name in _ROBOT_MODULES}
