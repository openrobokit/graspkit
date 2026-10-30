"""Tests for `scripts/benchmark/grasp_policy/package.py`."""

import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from graspkit.configs.experiment import build_experiment_config
from graspkit.models.checkpoint import load_policy
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.robot.grasp_robot import GraspRobot
from scripts.benchmark.grasp_policy.package import export_bundle


@pytest.mark.hf_assets
class TestPackage:
    @pytest.mark.parametrize("generator", ["cvae", "flow_transformer"])
    def test_native_package_roundtrip(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, generator: str):
        cfg = build_experiment_config("arx_x5_unified_depth")
        actor_cfg = replace(
            cfg.agent,
            vision_encoder="pointnet",
            vision_encoder_feature_dim=32,
            cvae_latent_dim=8,
            cvae_hidden_dim=32,
            generative_model=generator,
            normalize_cvae_actions=generator == "cvae",
            reconstruction_loss="phase_std_mse",
            grasp_evaluator_guidance_steps=0,
            grasp_evaluator_hand_joint_features=True,
            grasp_evaluator_mask_constant_features=False,
        )
        actor = build_grasp_policy(actor_cfg, cfg.robot_config, "cpu")
        actor.generator.action_center.copy_(torch.linspace(-0.1, 0.1, actor.generator.action_center.numel()))
        actor.generator.action_scale.fill_(0.3)
        actor_path = tmp_path / "actor.pt"
        torch.save(
            {
                "experiment": "arx_x5_unified_depth",
                "policy_config": asdict(actor_cfg),
                "agent": actor.state_dict(),
                "training": {"train_ids": ["training_scene"], "trainer": {"total_iters": 3}},
            },
            actor_path,
        )
        policy_cfg = replace(actor_cfg, grasp_evaluator_guidance_steps=1)
        policy = build_grasp_policy(policy_cfg, cfg.robot_config, "cpu")
        policy.load_state_dict({**policy.state_dict(), **actor.state_dict()}, strict=True)
        policy.grasp_evaluator.center.fill_(0.1)
        policy.grasp_evaluator.deviation.fill_(0.5)
        policy_path = tmp_path / "policy.pt"
        torch.save(
            {
                "experiment": "arx_x5_unified_depth",
                "policy_config": asdict(policy_cfg),
                "agent": policy.state_dict(),
                "iteration": 35,
                "training": {"iterations": 80},
            },
            policy_path,
        )
        dataset = replace(cfg.dataset, num_points=32, num_actions=4, return_object_pointcloud_only=False)
        (tmp_path / "recipe.json").write_text(json.dumps({"dataset": asdict(dataset)}))
        bundle = tmp_path / "bundle"
        manifest = export_bundle(policy_path, actor_path, bundle, {})
        assert json.loads((bundle / "bundle.json").read_text()) == manifest
        assert (bundle / "actor.pt").read_bytes() == actor_path.read_bytes()
        assert (bundle / "policy.pt").read_bytes() == policy_path.read_bytes()
        assert (manifest["actor_updates"], manifest["scorer_updates"], manifest["scorer_run_updates"]) == (4, 35, 80)
        conventions = json.loads((bundle / "config.json").read_text())
        helper = GraspRobot.from_config(cfg.robot_config)
        assert conventions["joint_order"] == helper.actuated_joint_names
        assert conventions["configuration"]["dataset"] == asdict(dataset)
        assert conventions["configuration"]["agent"] == asdict(policy_cfg)
        restored, restored_cfg, saved = load_policy(bundle / "policy.pt")
        assert restored_cfg.agent == policy_cfg
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in policy.state_dict().items())
        monkeypatch.chdir(bundle)
        observations = {"pointcloud": torch.randn(1, 32, 3), "qpos": torch.zeros(1, helper.num_actuated_joints)}
        code = (bundle / "README.md").read_text().split("```python\n")[1].split("```", 1)[0].replace("cuda:0", "cpu")
        example = tmp_path / "readme_example.py"
        example.write_text(code)
        namespace = runpy.run_path(str(example), init_globals={"observations": observations})
        assert torch.isfinite(namespace["actions"]).all()
