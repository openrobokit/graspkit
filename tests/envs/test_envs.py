"""Tests for `graspkit.envs.build_env` and its lazy backend imports."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("mani_skill")  # the maniskill extra

from graspkit import envs
from graspkit.configs.experiment_config import EnvConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import PROJECT_ROOT
from graspkit.envs import maniskill


class TestBuildEnv:
    def test_factory_does_not_import_unselected_backends_or_robot_assets(self, tmp_path: Path):
        code = """
import importlib.abc, sys
from unittest.mock import Mock

blocked = [
    'graspkit.envs.maniskill', 'graspkit.envs.mujoco', 'graspkit.envs.isaac', 'graspkit.envs.real',
    'graspkit.configs.mj_hand', 'robokit.assets.robots', 'mani_skill', 'graspkit.mani_skill',
    'mujoco', 'isaaclab', 'isaacsim', 'omni', 'xarm',
]

class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, *args: object):
        assert not fullname.startswith(tuple(blocked)), fullname

sys.meta_path.insert(0, Unavailable())
from graspkit.envs import build_env
from graspkit.configs.experiment_config import EnvConfig
from graspkit.constants import PROJECT_ROOT
from graspkit.configs.robot_config import RobotConfig

robot = RobotConfig(name='fixture', urdf_or_mjcf_path='unused.urdf')
assert build_env(None, robot, 'cpu') is None
assert not any(name.startswith(tuple(blocked)) for name in sys.modules)

blocked.remove('graspkit.envs.maniskill')
blocked.remove('mani_skill')
blocked.remove('graspkit.mani_skill')
blocked.remove('mujoco')  # ManiSkill's pytorch_kinematics dependency imports the SDK.
from graspkit.envs import MsEnv, maniskill
assert MsEnv is maniskill.MsEnv
factory = Mock()
maniskill.MsEnv = factory
cfg = EnvConfig()
assert build_env(cfg, robot, 'cpu', enable_rendering=False) is factory.return_value
factory.assert_called_once_with(cfg, robot_config=robot, device='cpu', renderer_cfg=None, enable_rendering=False)
assert not any(name.startswith(tuple(blocked)) for name in sys.modules)
print('backend import isolation: PASS')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=dict(
                os.environ,
                PYTHONPATH=f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
                CUDA_VISIBLE_DEVICES="",
                HF_HUB_OFFLINE="1",
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "backend import isolation: PASS" in result.stdout

    @pytest.mark.parametrize("name,backend", [("MsEnv", "maniskill"), ("MjEnv", "mujoco")])
    def test_public_backend_reexports_stay_lazy(self, tmp_path: Path, name: str, backend: str):
        code = """
import importlib.abc, sys, types
name, backend = sys.argv[1:]
selected = 'graspkit.envs.' + backend
module = types.ModuleType(selected)
expected = type(name, (), {})
setattr(module, name, expected)
sys.modules[selected] = module

class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, *args: object):
        assert not fullname.startswith((
            'graspkit.envs.maniskill', 'graspkit.envs.mujoco', 'graspkit.envs.isaac', 'graspkit.envs.real',
            'graspkit.configs.mj_hand', 'robokit.assets.robots',
            'mani_skill', 'mujoco', 'isaaclab', 'isaacsim', 'omni',
        )), fullname

sys.meta_path.insert(0, Unavailable())
namespace = {}
exec('from graspkit.envs import ' + name, namespace)
assert namespace[name] is expected
print('lazy public reexport: PASS')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, name, backend],
            cwd=tmp_path,
            env=dict(
                os.environ,
                PYTHONPATH=f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}",
                CUDA_VISIBLE_DEVICES="",
                HF_HUB_OFFLINE="1",
            ),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "lazy public reexport: PASS" in result.stdout

    @pytest.mark.parametrize(
        "options,rendering", [({}, True), ({"enable_rendering": True}, True), ({"enable_rendering": False}, False)]
    )
    def test_environment_factory_forwards_rendering_setting(
        self, monkeypatch: pytest.MonkeyPatch, options: dict, rendering: bool
    ):
        cfg = EnvConfig()
        robot = RobotConfig(name="fixture", urdf_or_mjcf_path="unused.urdf")
        device = torch.device("cpu")
        factory = Mock()
        monkeypatch.setattr(maniskill, "MsEnv", factory)
        assert envs.build_env(cfg, robot, device, **options) is factory.return_value
        factory.assert_called_once_with(
            cfg, robot_config=robot, device=device, renderer_cfg=None, enable_rendering=rendering
        )
