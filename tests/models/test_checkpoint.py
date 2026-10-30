"""Tests for `graspkit.models.checkpoint`."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from graspkit.configs.experiment import build_experiment_config
from graspkit.models.checkpoint import load_policy
from graspkit.models.grasp_policy import build_grasp_policy


@pytest.mark.hf_assets
class TestLoadPolicy:
    def test_native_checkpoint_roundtrip(self, tmp_path: Path):
        cfg = build_experiment_config("arx_x5_unified_depth")
        agent_cfg = replace(
            cfg.agent,
            vision_encoder_feature_dim=32,
            cvae_latent_dim=8,
            cvae_hidden_dim=32,
            normalize_cvae_actions=True,
            reconstruction_loss="phase_std_mse",
            grasp_evaluator_guidance_steps=2,
            grasp_evaluator_hand_joint_features=True,
            grasp_evaluator_mask_constant_features=False,
        )
        policy = build_grasp_policy(agent_cfg, cfg.robot_config).eval().requires_grad_(False)
        policy.generator.action_scale.fill_(0.2)
        saved = {"experiment": cfg.name, "policy_config": asdict(agent_cfg), "agent": policy.state_dict()}
        path = tmp_path / "policy.pt"
        torch.save(saved, path)
        loaded, loaded_cfg, _ = load_policy(path)
        assert loaded_cfg.agent == agent_cfg
        obs = {"pointcloud": torch.randn(1, 32, 3)}
        torch.manual_seed(7)
        reference = policy.get_action(obs, 4)
        torch.manual_seed(7)
        torch.testing.assert_close(loaded.get_action(obs, 4), reference, atol=0, rtol=0)

    def test_stock_native_checkpoint_keeps_defaults_without_new_flags(self, tmp_path: Path):
        cfg = build_experiment_config("arx_x5_unified_depth")
        agent_cfg = replace(
            cfg.agent,
            vision_encoder_feature_dim=32,
            cvae_latent_dim=8,
            cvae_hidden_dim=32,
            grasp_evaluator_guidance_steps=2,
        )
        policy = build_grasp_policy(agent_cfg, cfg.robot_config).eval().requires_grad_(False)
        flags = {
            "normalize_cvae_actions",
            "reconstruction_loss",
            "grasp_evaluator_hand_joint_features",
            "grasp_evaluator_mask_constant_features",
        }
        saved = {
            "experiment": cfg.name,
            "policy_config": {k: v for k, v in asdict(agent_cfg).items() if k not in flags},
            "agent": policy.state_dict(),
        }
        path = tmp_path / "stock.pt"
        torch.save(saved, path)
        loaded, loaded_cfg, _ = load_policy(path)
        assert loaded_cfg.agent == agent_cfg
        assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in policy.state_dict().items())
