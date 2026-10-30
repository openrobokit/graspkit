"""Tests for the Sharpa experiments in `graspkit.configs.experiment.sharpa`."""

from pathlib import Path

import pytest

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.robot import build_robot_config


@pytest.mark.hf_assets
class TestSharpaExperiment:
    def test_sharpa_left_experiment(self):
        pytest.importorskip("mani_skill")  # the maniskill extra
        from robokit.assets.robots.end_effectors import sharpa_hand

        from graspkit.mani_skill.agents.sharpa import SharpaLeftFloatingBase, SharpaRightFloatingBase

        left = build_experiment_config("sharpa_left_dex1b_v0_5_mini_tabletop_gd_zed")
        right = build_experiment_config("sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed")
        robot = build_robot_config("sharpa_left")
        assert left.robot_config == robot
        assert robot.base_type == "floating"
        assert robot.palm_link_name == "left_hand_C_MC"
        assert robot.contact_points_path == str(sharpa_hand.CONTACT_POINTS_DIR_LEFT / "contact_points.json")
        assert robot.self_collision_ignore_path == str(sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT)
        assert left.traj_builder.floating_base_robot_config == robot
        assert left.traj_builder.fixed_base_robot_config.name == "vega_sharpa_left"
        assert left.grasp_optim.evaluator == left.trainer.evaluator
        for cfg, uid in ((left, SharpaLeftFloatingBase.uid), (right, SharpaRightFloatingBase.uid)):
            assert cfg.grasp_optim.evaluator.ms_env_config.robot_uid == uid
            assert cfg.renderer.env_config.ms_env_config.robot_uid == uid
            assert cfg.trainer.evaluator.ms_env_config.robot_uid == uid
        assert left.renderer.camera_poses_path == right.renderer.camera_poses_path
        assert Path(left.renderer.camera_poses_path).is_file()
        for field in ("grasp_output_dir", "rendered_data_path", "processed_data_path"):
            assert getattr(left, field) != getattr(right, field)
