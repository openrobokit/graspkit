"""Tests for `graspkit.envs.base.BaseEnv` and the deploy contract its adapters implement."""

import inspect

import numpy as np
import pytest
import torch

pytest.importorskip("mani_skill")  # the maniskill extra

from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv
from graspkit.envs.maniskill import MsEnv
from graspkit.scene.scene_description import SceneDescription


class FakeEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.commanded: list[torch.Tensor] = []

    def reset(
        self,
        scene: SceneDescription | None = None,
        init_qpos: torch.Tensor | None = None,
        root_pose: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ):
        self.commanded = []

    def get_qpos(self) -> torch.Tensor:
        # stand-in for controller lag: measured trails the last command, so it is distinguishable from it
        return torch.zeros((1, 2)) if not self.commanded else self.commanded[-1].reshape(1, 2) * 0.5

    def step_qpos(self, q: torch.Tensor):
        self.commanded.append(q.clone())

    def get_info(self) -> dict[str, torch.Tensor]:
        return {"steps": torch.tensor(len(self.commanded))}

    def get_rgb(self) -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.uint8)

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        return torch.zeros((1, 4, 3))

    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def close(self):
        self.commanded = []


class TestBaseEnv:
    def test_base_env_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEnv()
        assert BaseEnv.__abstractmethods__ == {
            "reset",
            "get_qpos",
            "step_qpos",
            "get_info",
            "get_rgb",
            "get_pointcloud",
            "evaluate",
            "close",
        }
        assert issubclass(MsEnv, BaseEnv)
        assert not MsEnv.__abstractmethods__  # MsEnv is concrete

    def test_set_qpos_traj_streams_measured_qpos_every_step(self):
        env = FakeEnv()
        traj = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        visualized = []

        infos = env.set_qpos_traj(traj, lambda step, qpos: visualized.append((step, qpos.clone())))

        torch.testing.assert_close(torch.stack(env.commanded), traj)
        assert [step for step, _ in visualized] == list(range(len(traj)))
        torch.testing.assert_close(torch.stack([qpos for _, qpos in visualized]), traj * 0.5)
        assert infos["steps"].item() == len(traj)

    @pytest.mark.hf_assets
    def test_sim_env_adapters_implement_deploy_contract(self):
        from graspkit.envs.mujoco import MjEnv

        assert not inspect.isabstract(MsEnv)
        assert not inspect.isabstract(MjEnv)

    def test_sim_env_deploy_methods_exist(self):
        """Cheap structural check that deploy primitives exist (full rollout needs scene assets; see deploy.py)."""
        for name in ("reset", "get_qpos", "step_qpos", "get_info", "get_rgb", "get_pointcloud", "evaluate"):
            assert callable(getattr(MsEnv, name))
