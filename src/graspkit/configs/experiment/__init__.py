"""ExperimentConfig registry: name -> module basename (explicit; no discovery, no import-time fetch)."""

import importlib
from typing import Dict

from graspkit.configs.experiment_config import ExperimentConfig

_EXPERIMENT_MODULES: Dict[str, str] = {
    "ability_right_dex1b_v0_4_mini_floating": "ability",
    "ability_right_dex1b_v0_4_mini_tabletop": "ability",
    "agilex_aloha_tabletop": "grasp_policy",
    "arx_x5_tabletop": "grasp_policy",
    "panda_allegro_tabletop": "grasp_policy",
    "yam_tabletop": "grasp_policy",
    "xarm7_sharpa_tabletop": "grasp_policy",
    "arx_x5_unified_depth": "grasp_policy",
    "arx_x5_unified_rgb": "grasp_policy",
    "panda_allegro_unified_depth": "grasp_policy",
    "panda_allegro_unified_rgb": "grasp_policy",
    "agilex_aloha_unified_depth": "grasp_policy",
    "agilex_aloha_unified_rgb": "grasp_policy",
    "yam_unified_depth": "grasp_policy",
    "yam_unified_rgb": "grasp_policy",
    "xarm7_sharpa_unified_depth": "grasp_policy",
    "xarm7_sharpa_unified_rgb": "grasp_policy",
    "agilex_aloha_arm_mj_tabletop": "tabletop_jaw_mj",
    "arx_x5_arm_mj_tabletop": "tabletop_jaw_mj",
    "arx_x5_gripper_dex1b_mini_tabletop": "arx_x5_gripper",
    "arx_x5_diverse": "arx_x5_gripper",
    "arx_x5_diverse_pre": "arx_x5_gripper",
    "arx_x5_diverse_ft": "arx_x5_gripper",
    "g1_inspire_right_dex1b_mini_tabletop": "g1_inspire",
    "g1_inspire_right_dex1b_v0_4_mini_tabletop": "g1_inspire",
    "g1_inspire_right_dex1b_v0_4_dev": "g1_inspire",
    "g1_inspire_right_wbc_dex1b_v0_4_1_mini": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_4_1_mini_s3": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_small": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_small_s3": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_mini": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_mini_s3": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_mini_pen": "g1_inspire_wbc",
    "g1_inspire_right_wbc_dex1b_v0_5_mini_pen_samefilter": "g1_inspire_wbc",
    "inspire_right_dex1b_v0_4_mini_floating": "inspire",
    "inspire_right_dex1b_v0_4_mini_tabletop": "inspire",
    "inspire_right_dex1b_v0_4_dev": "inspire",
    "inspire_right_dex1b_v0_5_mini_tabletop": "inspire",
    "mano_right_dex1b_v0_4_mini_floating": "mano",
    "mano_right_dex1b_v0_4_mini_tabletop": "mano",
    "g1_mano_right_dex1b_v0_4_mini_tabletop": "mano",
    "shadow_right_mj_val": "shadow_mj",
    "shadow_right_mj_val_tabletop": "shadow_mj",
    "paxini_right_dex1b_v0_4_mini_floating": "paxini",
    "panda_allegro_right_mj_tabletop": "panda_allegro_mj",
    "sharpa_right_dex1b_v0_4_mini_floating": "sharpa",
    "sharpa_right_dex1b_v0_5_mini_tabletop_gd": "sharpa",
    "sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed": "sharpa",
    "sharpa_left_dex1b_v0_5_mini_tabletop_gd_zed": "sharpa",
    "sharpa_left_plain": "sharpa_left_plain",
    "vega_sharpa_right_dex1b_v0_5_mini_zed": "vega_sharpa",
    "vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop": "vega_sharpa_mj",
    "vega_sharpa_left_mj_dex1b_v0_5_mini_tabletop": "vega_sharpa_mj",
    "xarm7_shadow_right_dex1b_v0_4_mini_tabletop": "xarm7_shadow",
    "xarm7_xhand_right_mj_tabletop": "xarm7_xhand_mj",
    "xarm7_ability_right_centered_pointnet": "xarm7_ability",
    "xarm7_ability_right_centered_spconv_layernorm_local_01kl": "xarm7_ability",
    "xarm7_xhand_right_dex1b_v0_4_mini_tabletop": "xarm7_xhand",
    "xarm7_xhand_right_dex1b_v0_5_mini_tabletop": "xarm7_xhand",
    "xarm7_xhand_right_dex1b_v0_5_mini_tabletop_centered": "xarm7_xhand",
    "xhand_right_mj_dex1b_v0_5_tabletop": "xhand_mj",
    "xhand_right_dex1b_v0_4_mini_floating": "xhand",
    "xhand_right_dex1b_v0_4_mini_tabletop": "xhand",
    "xhand_right_dex1b_v0_5_mini_tabletop": "xhand",
    "xhand_right_dex1b_v0_5_mini_tabletop_fixed_eval": "xhand",
}


def build_experiment_config(name: str) -> ExperimentConfig:
    if name not in _EXPERIMENT_MODULES:
        raise KeyError(f"unknown experiment '{name}'; known: {sorted(_EXPERIMENT_MODULES)}")
    return getattr(importlib.import_module(f"graspkit.configs.experiment.{_EXPERIMENT_MODULES[name]}"), name)


def cli_experiment() -> ExperimentConfig:
    """Import only the named experiment from argv (so only its assets download), then apply tyro overrides to it."""
    import sys

    import tyro

    argv = sys.argv[1:]
    if not argv or argv[0] not in _EXPERIMENT_MODULES:
        listing = "\n  ".join(sorted(_EXPERIMENT_MODULES))
        print(f"usage: <experiment_name> [overrides]\n\nexperiments:\n  {listing}")
        raise SystemExit(0 if argv[:1] in (["-h"], ["--help"]) else 2)
    return tyro.cli(ExperimentConfig, default=build_experiment_config(argv[0]), args=argv[1:], use_underscores=True)
