"""Tests for `graspkit.envs.isaac.IsaacEnv`."""

import inspect
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from graspkit.configs.env_config import IsaacEnvConfig, MjEvalConfig
from graspkit.configs.experiment_config import EnvConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.envs import build_env
from graspkit.envs.base import BaseEnv
from graspkit.envs.isaac import IsaacEnv


class TestIsaacEnv:
    def test_isaac_factory_and_runtime_isolation(self, monkeypatch: pytest.MonkeyPatch):
        assert issubclass(IsaacEnv, BaseEnv) and not inspect.isabstract(IsaacEnv)
        assert "isaaclab" not in sys.modules
        factory = Mock(return_value="isaac")
        monkeypatch.setattr("graspkit.envs.isaac.IsaacEnv", factory)
        config = EnvConfig(isaac_env_config=IsaacEnvConfig())
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf")
        assert build_env(config, robot, "cpu") == "isaac"
        factory.assert_called_once_with(config, robot, "cpu", None)
        with pytest.raises(AssertionError, match="Select one"):
            build_env(EnvConfig(isaac_env_config=IsaacEnvConfig(), mj_env_config=MjEvalConfig()), robot, "cpu")
        saved = asdict(config)
        saved.pop("isaac_env_config")
        assert EnvConfig(**saved).isaac_env_config is None

    def test_evaluation_chunks_and_seed_successes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        env = IsaacEnv.__new__(IsaacEnv)
        env.cfg = EnvConfig(num_envs=2, num_seeds=2, enable_perturbation=False)
        env.isaac = IsaacEnvConfig()
        env.directory, env.device, env.attrs = tmp_path, torch.device("cpu"), {}
        env.robot = {
            "root_pose": np.asarray([0, 0, 0, 1, 0, 0, 0]),
            "coefficients": np.eye(2, dtype=np.float32),
            "bias": np.zeros(2, np.float32),
        }
        monkeypatch.setattr("graspkit.envs.isaac.write_job", Mock())
        env._request = Mock(
            side_effect=[
                {"success": np.asarray([True, True]), "lift_dz": np.asarray([0.3, 0.4], np.float32)},
                {"success": np.asarray([True]), "lift_dz": np.asarray([0.5], np.float32)},
                {"success": np.asarray([True, False]), "lift_dz": np.asarray([0.3, 0.1], np.float32)},
                {"success": np.asarray([True]), "lift_dz": np.asarray([0.5], np.float32)},
            ]
        )
        batch = SimpleNamespace(full_traj_qpos=torch.zeros(3, 1, 4, 2), scene_descriptions=[None] * 3)
        actual = env.evaluate(batch)
        torch.testing.assert_close(actual["success"], torch.tensor([[True], [False], [True]]))
        torch.testing.assert_close(actual["lift_dz"], torch.tensor([[0.3], [0.1], [0.5]]))
        assert [call.kwargs["indices"] for call in env._request.call_args_list] == [[0, 1], [2], [0, 1], [2]]
        assert all(call.kwargs["forces"] is None for call in env._request.call_args_list)
