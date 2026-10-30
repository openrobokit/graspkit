"""Tests for `scripts/stages/train/train.py`."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import torch

from graspkit.configs.experiment_config import DatasetConfig, ExperimentConfig, TrainerConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.grasp_policy import GraspPolicy
from graspkit.scene.scene_description import SceneDescription
from scripts.stages.train import train as train_module


class TestTrain:
    @pytest.mark.parametrize("cache_data", [False, True])
    def test_training_entrypoint_omits_eval_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_data: bool
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "set_device", Mock())
        dataset = [0, 1]
        dataset_factory = Mock(return_value=Mock(scene_ids=["train"], __len__=Mock(return_value=len(dataset))))
        monkeypatch.setattr(train_module, "GraspDataset", dataset_factory)
        loader = Mock()
        monkeypatch.setattr(train_module, "get_iteration_based_dataloader", Mock(return_value=loader))
        trainer = Mock()
        monkeypatch.setattr(train_module, "Trainer", Mock(return_value=trainer))
        cfg = ExperimentConfig(
            name="test", agent=GraspPolicyConfig(), dataset=DatasetConfig(cache_in_memory=cache_data)
        )
        train_module.train(cfg)
        dataset_factory.assert_called_once_with(
            replace(cfg.dataset, return_rgb=False),
            h5_path=cfg.processed_data_path,
            scene_description_path=cfg.scene_description_path,
            split="train",
            indices_only=False,
        )
        trainer.train.assert_called_once_with(train_dataloader=loader, test_dataloader=None)
        trainer.writer.close.assert_called_once_with()
        assert Path("logs/train/test/.train_complete").is_file()

    def test_scratch_flow_normalization_with_cached_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(torch.cuda, "set_device", Mock())
        SceneDescription.save_scene_descriptions_msgpack(
            {"train": SceneDescription(scene_id="train")}, "scenes.msgpack"
        )
        center = np.arange(24, dtype=np.float32).reshape(3, 8)
        scale = np.full((3, 8), 0.5, dtype=np.float32)
        with h5py.File("train.h5", "w") as data:
            data["meta/action_mean"] = center
            data["meta/action_std"] = scale
            data["data/train/actions"] = np.zeros((2, 3, 8), dtype=np.float32)
            data["data/train/obs/pointcloud"] = np.zeros((1, 4, 3), dtype=np.float32)
            data["data/train/obs/pointcloud_segmentation"] = np.ones((1, 4, 1), dtype=np.int32)
            data["data/train/obs/target_actor_segmentation_id"] = 1
        agent_cfg = GraspPolicyConfig(vision_encoder="pointnet", generative_model="flow_transformer", cvae_hidden_dim=8)
        policy = GraspPolicy(agent_cfg, 8, RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="fixed"))
        trainer = Mock(agent=policy)
        monkeypatch.setattr(train_module, "Trainer", Mock(return_value=trainer))
        cfg = ExperimentConfig(
            name="scratch_flow",
            agent=agent_cfg,
            dataset=DatasetConfig(
                num_actions=1, num_points=4, return_object_pointcloud_only=True, cache_in_memory=True
            ),
            trainer=TrainerConfig(batch_size=1, num_dataloader_workers=0),
            scene_description_path="scenes.msgpack",
            processed_data_path="train.h5",
        )
        train_module.train(cfg)
        torch.testing.assert_close(policy.generator.action_center, torch.from_numpy(center).flatten())
        torch.testing.assert_close(policy.generator.action_scale, torch.from_numpy(scale).flatten())
        trainer.train.assert_called_once()
