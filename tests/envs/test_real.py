"""Tests for `graspkit.envs.real.RealEnv` that do not open hardware."""

import pytest

pytest.importorskip("mani_skill")  # the maniskill extra

from graspkit.envs import BaseEnv, MsEnv


class TestRealEnv:
    def test_real_env_subclass(self):
        pytest.importorskip("retargetingkit", reason="RealEnv needs the xarm extra")
        from graspkit.envs import RealEnv  # do NOT construct — that opens hardware

        assert issubclass(RealEnv, BaseEnv)
        assert not RealEnv.__abstractmethods__  # concrete
        public = {n for n in ("reset", "get_qpos", "step_qpos", "get_info", "get_rgb", "get_pointcloud", "evaluate")}
        assert public <= set(dir(RealEnv)) and public <= set(dir(MsEnv))

    def test_real_env_uses_shared_masked_depth_projection(self):
        pytest.importorskip("retargetingkit", reason="RealEnv needs the xarm extra")
        import graspkit.envs.real as real
        import graspkit.utils.point_cloud_utils as pcu

        assert real.masked_depth_to_point_cloud is pcu.masked_depth_to_point_cloud
