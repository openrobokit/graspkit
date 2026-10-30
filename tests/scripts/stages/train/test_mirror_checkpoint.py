"""Tests for `scripts/stages/train/mirror_checkpoint.py`."""

from copy import deepcopy

import pytest
import torch
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix

from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.grasp_policy import GraspPolicy
from scripts.stages.train.mirror_checkpoint import mirror_floating_pointnet_model


class TestMirrorCheckpoint:
    @pytest.mark.parametrize("rot6d", [False, True])
    def test_mirrored_pointnet_policy(self, rot6d: bool):
        torch.manual_seed(0)
        cfg = GraspPolicyConfig(
            vision_encoder="pointnet", vision_encoder_feature_dim=32, convert_rxyz_to_rot6d=rot6d, use_mean=False
        )
        robot = RobotConfig(name="test", urdf_or_mjcf_path="unused.urdf", base_type="floating")
        right = GraspPolicy(cfg, 31 if rot6d else 28, robot).eval()
        left = deepcopy(right)
        mirror_floating_pointnet_model(left)
        points = torch.randn(3, 32, 3)
        reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        torch.manual_seed(3)
        right_actions = right.get_action({"pointcloud": points @ reflection}, num_samples_per_scene=4)
        torch.manual_seed(3)
        left_actions = left.get_action({"pointcloud": points}, num_samples_per_scene=4)
        torch.testing.assert_close(left_actions[..., :3], right_actions[..., :3] @ reflection)
        torch.testing.assert_close(left_actions[..., 6:], right_actions[..., 6:])
        torch.testing.assert_close(
            euler_angles_to_matrix(left_actions[..., 3:6], "rxyz"),
            reflection @ euler_angles_to_matrix(right_actions[..., 3:6], "rxyz") @ reflection,
            atol=1e-5,
            rtol=1e-5,
        )
        actions = torch.randn(3, 4, 3, 28) * 0.3
        mirrored_actions = actions.clone()
        mirrored_actions[..., [1, 3, 5]] *= -1
        torch.manual_seed(7)
        right_losses = right({"pointcloud": points @ reflection}, mirrored_actions)
        torch.manual_seed(7)
        left_losses = left({"pointcloud": points}, actions)
        for key in right_losses:
            torch.testing.assert_close(left_losses[key], right_losses[key])
        mirror_floating_pointnet_model(left)
        for key, value in right.state_dict().items():
            torch.testing.assert_close(left.state_dict()[key], value, atol=0, rtol=0)
