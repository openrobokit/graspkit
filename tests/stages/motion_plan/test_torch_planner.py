"""Tests for `graspkit.stages.motion_plan.torch_planner`."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.stages.motion_plan.torch_planner import TorchPlanner


class TestTorchPlanner:
    @pytest.mark.parametrize("shape", [(1, 1), (4, 1), (2, 3)])
    def test_lift_ik_receives_only_real_targets_and_preserves_layout(self, shape: tuple[int, int]):
        qpos = torch.arange(shape[0] * shape[1] * 4, dtype=torch.float32).reshape(*shape, 4) / 10
        poses = torch.eye(4).expand(*shape, 1, 4, 4).clone()
        poses[..., 0, :3, 3] = qpos[..., :3]
        arm_indices = torch.tensor([0, 2])
        inverse_kinematics = Mock(return_value=(qpos[..., arm_indices].reshape(-1, 2) + 1, torch.ones(shape)))
        helper = SimpleNamespace(
            forward_kinematics_torch=Mock(return_value=poses),
            link_names=["hand"],
            arm_joint_indices=arm_indices,
            root_pose_mat=torch.eye(4),
            inverse_kinematics=inverse_kinematics,
        )
        planner = TorchPlanner.__new__(TorchPlanner)
        planner.cfg = MotionPlannerConfig(
            lifting_dist=0.3, num_samples_per_batch=2048, start_from_grasp_dof_indices=(3,)
        )
        planner.robot_config = SimpleNamespace(base_type="fixed", hand_base_link_name="hand", arm_base_link_name=None)
        planner.robot_helper = helper
        actual = planner.get_lifting_target_qpos(qpos)
        inverse_kinematics.assert_called_once()
        call = inverse_kinematics.call_args
        targets = call.args[0]
        assert targets.shape == (shape[0] * shape[1], 4, 4)
        expected_targets = poses[..., 0, :, :].reshape(-1, 4, 4).clone()
        expected_targets[:, 2, 3] += 0.3
        torch.testing.assert_close(targets, expected_targets)
        torch.testing.assert_close(call.kwargs["init_qpos"], qpos[..., arm_indices].reshape(-1, 2))
        assert call.kwargs["init_sample_range"] == 0.1
        assert call.kwargs["locked_joint_indices"] == (3,)
        expected = qpos.clone()
        expected[..., arm_indices] += 1
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(qpos, torch.arange(qpos.numel(), dtype=torch.float32).reshape(*shape, 4) / 10)
