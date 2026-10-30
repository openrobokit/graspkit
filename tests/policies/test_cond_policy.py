"""Tests for `graspkit.policies.cond_policy`."""

from functools import partial

import pytest
import torch

from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot import build_robot_config
from graspkit.models.grasp_policy import build_grasp_policy


@pytest.mark.hf_assets
class TestCondGraspPolicy:
    @pytest.mark.parametrize("version,local_k", [(0, 0), (2, 0), (2, 4)])
    def test_conditional_factory_and_requested_point_translation(self, version: int, local_k: int):
        from graspkit.policies.cond_policy import COND_DIM, V2_DIM, CondGraspPolicy  # imports robot assets

        cfg = GraspPolicyConfig(
            vision_encoder="pointnet",
            vision_encoder_feature_dim=8,
            center_pointcloud=True,
            cvae_hidden_dim=8,
            cvae_latent_dim=2,
            use_mean=False,
        )
        cls = partial(CondGraspPolicy, version=version, local_k=local_k)
        policy = build_grasp_policy(cfg, build_robot_config("arx_x5_gripper_robodojo"), policy_cls=cls).eval()
        assert isinstance(policy, CondGraspPolicy)
        points = torch.rand(2, 12, 3) * 0.1
        condition = torch.zeros(2, V2_DIM if version == 2 else COND_DIM)
        if version == 2:
            condition[:, :3], condition[:, 7] = points[:, 0], 1
        else:
            condition[:, :3], condition[:, 3:6], condition[:, 15] = points[:, 0], points[:, 1], 1
        offset = torch.tensor([0.2, -0.1, 0.3])
        shifted = condition.clone()
        shifted[:, :3] += offset
        if version == 0:
            shifted[:, 3:6] += offset
        with torch.inference_mode():
            torch.manual_seed(3)
            expected = policy.get_action({"pointcloud": points, "condition": condition}, 2)
            torch.manual_seed(3)
            actual = policy.get_action({"pointcloud": points + offset, "condition": shifted}, 2)
        assert expected.shape == (2, 2, 3, 7) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual[..., :3], expected[..., :3] + offset, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[..., 3:], expected[..., 3:], atol=1e-6, rtol=1e-6)
        assert any(key.startswith("cvae.") for key in policy.state_dict())
