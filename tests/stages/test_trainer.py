"""Tests for `graspkit.stages.trainer`."""

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from graspkit.configs.experiment_config import EnvConfig, ExperimentConfig, TrainerConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.grasp_policy import GraspPolicy
from graspkit.stages import trainer as trainer_module
from scripts.stages.train.train import get_iteration_based_dataloader


class RepeatedBatch(list):
    scene_ids = ("train", "train")


class TestTrainer:
    def test_training_without_evaluator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        scene_load = Mock(side_effect=AssertionError("Training must not load evaluation scenes"))
        env_build = Mock(side_effect=AssertionError("Training must not construct an evaluator"))
        monkeypatch.setattr(trainer_module.SceneDescription, "load_scene_descriptions_msgpack", scene_load)
        monkeypatch.setattr(trainer_module, "build_env", env_build)
        policy_cfg = GraspPolicyConfig(vision_encoder="pointnet", vision_encoder_feature_dim=32, cvae_latent_dim=8)
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        agent = GraspPolicy(policy_cfg, 28, robot)
        monkeypatch.setattr(trainer_module, "build_grasp_policy", Mock(return_value=agent))
        cfg = ExperimentConfig(
            name="plain_test",
            agent=policy_cfg,
            robot_config=robot,
            trainer=TrainerConfig(evaluator=None, total_iters=3, eval_freq=1, save_freq=1),
        )
        trainer = trainer_module.Trainer(cfg)
        evaluate = Mock(side_effect=AssertionError("Training must not evaluate"))
        monkeypatch.setattr(trainer, "evaluate", evaluate)
        before = {key: tensor.clone() for key, tensor in agent.state_dict().items()}
        batch = {"observations": {"pointcloud": torch.randn(2, 32, 3)}, "actions": torch.randn(2, 4, 3, 28)}
        trainer.train([batch, batch])
        trainer.writer.close()
        scene_load.assert_not_called()
        env_build.assert_not_called()
        evaluate.assert_not_called()
        assert any(not torch.equal(before[key], tensor) for key, tensor in agent.state_dict().items())
        initial = torch.load("logs/train/plain_test/checkpoints/initial.pt", weights_only=True)["agent"]
        assert all(torch.equal(tensor, initial[key]) for key, tensor in before.items())
        assert Path("logs/train/plain_test/checkpoints/iter_1.pt").is_file()

    def test_evaluator_is_created_only_when_requested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "set_device", Mock())
        monkeypatch.setattr(trainer_module, "build_grasp_policy", Mock(return_value=torch.nn.Linear(2, 2)))
        scenes = Mock(return_value={})
        builder = Mock(return_value=SimpleNamespace(robot=SimpleNamespace(config=None)))
        environment = Mock()
        monkeypatch.setattr(trainer_module.SceneDescription, "load_scene_descriptions_msgpack", scenes)
        monkeypatch.setattr(trainer_module, "GraspTrajBuilder", builder)
        monkeypatch.setattr(trainer_module, "build_env", environment)
        cfg = ExperimentConfig(
            name="eval_test",
            agent=GraspPolicyConfig(),
            trainer=TrainerConfig(evaluator=EnvConfig(), batch_size=4, eval_num_envs=8),
        )
        trainer = trainer_module.Trainer(cfg)
        scenes.assert_not_called()
        builder.assert_not_called()
        environment.assert_not_called()
        assert trainer.evaluate([]) == {"scene_ids": []}
        assert cfg.trainer.evaluator.num_envs == 4
        scenes.assert_called_once_with(cfg.scene_description_path)
        builder.assert_called_once_with(cfg.traj_builder, trainer.device)
        environment.assert_called_once_with(cfg.trainer.evaluator, None, trainer.device)
        trainer.writer.close()

    def test_native_trainer_resume_publication(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        policy_cfg = GraspPolicyConfig(vision_encoder="pointnet", vision_encoder_feature_dim=32, cvae_latent_dim=8)
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        cfg = ExperimentConfig(
            name="resume_cpu",
            agent=policy_cfg,
            robot_config=robot,
            trainer=TrainerConfig(
                evaluator=None, total_iters=3, batch_size=2, num_dataloader_workers=0, save_freq=2, log_freq=1
            ),
        )
        torch.manual_seed(41)
        initial = GraspPolicy(policy_cfg, 28, robot).state_dict()
        sample = {"observations": {"pointcloud": torch.randn(16, 3)}, "actions": torch.randn(4, 3, 28)}

        def build(*args: object) -> GraspPolicy:
            policy = GraspPolicy(policy_cfg, 28, robot)
            policy.load_state_dict(initial, strict=True)
            return policy

        monkeypatch.setattr(trainer_module, "build_grasp_policy", build)
        metadata = {
            "experiment": "resume_cpu",
            "policy_config": {"recipe": "test"},
            "training": {"train_ids": ["train"]},
        }
        published = []

        def publish(path: Path):
            pointer = json.loads((path / "latest.json").read_text())
            published.append(pointer["updates"])

        def run(count: int, path: Path, resume: dict | None = None) -> dict:
            seed = 53 if resume is None else 97
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            trainer = trainer_module.Trainer(cfg)
            loader = get_iteration_based_dataloader(
                RepeatedBatch([sample, sample]),
                replace(cfg, trainer=replace(cfg.trainer, total_iters=count - 1)),
                generator=torch.Generator().manual_seed(71),
            )
            trainer.train(loader, checkpoint_dir=path, metadata=metadata, resume=resume, publish=publish)
            trainer.writer.close()
            return torch.load(path / "latest.pt", weights_only=True)

        uninterrupted = run(4, tmp_path / "full")
        partial = run(2, tmp_path / "split")
        resumed = run(2, tmp_path / "split", partial)
        assert published == [2, 4, 2, 4]
        for key in ("agent", "optimizer", "scheduler", "torch_rng"):
            torch.testing.assert_close(uninterrupted[key], resumed[key], rtol=0, atol=0)
        assert uninterrupted["python_rng"] == resumed["python_rng"]
        assert uninterrupted["numpy_rng"][0] == resumed["numpy_rng"][0]
        torch.testing.assert_close(uninterrupted["numpy_rng"][1], resumed["numpy_rng"][1], rtol=0, atol=0)
        assert uninterrupted["numpy_rng"][2:] == resumed["numpy_rng"][2:]
        assert resumed["iteration"] == 3 and resumed["training"] == metadata["training"]

    def test_finite_gradient_overflow_is_published(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class OverflowPolicy(torch.nn.Linear):
            def forward(self, observations: dict, actions: torch.Tensor) -> dict:
                return {"loss": self.weight.sum() * 1e38}

        policy = OverflowPolicy(2, 1, bias=False).float()
        monkeypatch.setattr(trainer_module, "build_grasp_policy", lambda *args: policy)
        cfg = ExperimentConfig(
            name="overflow_cpu",
            agent=GraspPolicyConfig(),
            trainer=TrainerConfig(
                total_iters=3, batch_size=2, num_dataloader_workers=0, evaluator=None, fast_step=False, grad_clip_norm=1
            ),
        )
        trainer = trainer_module.Trainer(cfg)
        batch = {"observations": {"pointcloud": torch.zeros(1, 3)}, "actions": torch.zeros(1)}
        loader = get_iteration_based_dataloader(
            RepeatedBatch([batch, batch]),
            cfg,
            generator=torch.Generator().manual_seed(0),
        )
        before = policy.weight.detach().clone()
        scheduler = trainer.lr_scheduler.state_dict()
        output = tmp_path / "failure"
        published = []
        with pytest.raises(FloatingPointError):
            trainer.train(loader, checkpoint_dir=output, metadata={"recipe": "test"}, publish=published.append)
        trainer.writer.close()
        assert torch.isfinite(policy.weight.grad).all() and not torch.isfinite(policy.weight.grad.norm())
        assert torch.equal(before, policy.weight) and not trainer.optimizer.state
        assert trainer.lr_scheduler.state_dict() == scheduler
        assert published == [output] and not (output / "latest.pt").exists()
        diagnostic = json.loads((output / "gradient_failure.json").read_text())
        assert diagnostic["nonfinite_gradients"] == [] and diagnostic["gradient_norms"]["weight"] == "inf"
        assert torch.load(output / "gradient_failure.pt", weights_only=True)["recipe"] == "test"
