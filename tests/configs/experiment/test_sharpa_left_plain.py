"""Tests for `graspkit.configs.experiment.sharpa_left_plain`."""

import pytest

from graspkit.configs.experiment import build_experiment_config


@pytest.mark.hf_assets
class TestSharpaLeftPlain:
    def test_plain_left_config(self):
        cfg = build_experiment_config("sharpa_left_plain")
        assert cfg.robot_config.name == "sharpa_left"
        assert cfg.steps == ("train",) and cfg.ckpt_path is None
        assert cfg.trainer.evaluator is None
        assert cfg.trainer.seed == 29 and cfg.trainer.total_iters == 200000
        assert cfg.trainer.save_freq == 40000 and cfg.agent.kl_weight == 0.5
        assert cfg.agent.vision_encoder == "pointnet"
        assert not cfg.agent.cvae_learned_prior and not cfg.agent.coupled_floating_base
        assert not cfg.agent.canonical_yaw and not cfg.agent.enable_penetration_loss
        assert cfg.agent.grasp_evaluator_guidance_steps == 0 and cfg.agent.prior_fingertip_weight == 0
