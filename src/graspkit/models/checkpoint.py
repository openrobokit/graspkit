"""Strict policy loading: the saved policy_config and experiment rebuild the exact network."""

from dataclasses import replace
from pathlib import Path

import torch

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy


def load_policy(path: Path, device: str = "cpu") -> tuple[GraspPolicy, ExperimentConfig, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    policy_cfg = GraspPolicyConfig(**saved["policy_config"])
    cfg = replace(build_experiment_config(saved["experiment"]), agent=policy_cfg)
    policy = build_grasp_policy(policy_cfg, cfg.robot_config, "cpu")
    policy.load_state_dict(saved["agent"], strict=True)
    return policy.to(device).eval().requires_grad_(False), cfg, saved
