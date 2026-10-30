"""Tests for `graspkit.stages.motion_plan.warp_planner`."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.scene.scene_description import BoxDescription, SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.motion_plan.warp_planner import WarpPlanner


class TestWarpPlanner:
    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("missing", [True, False])
    def test_motion_planner_sphere_ignores_and_warning(
        self,
        collision_config: RobotConfig,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        enabled: bool,
        missing: bool,
    ):
        # Point at the capsule file: trajectory terms must still select sphere ignores.
        collision_config = replace(
            collision_config,
            self_collision_ignore_path=str(
                Path(collision_config.self_collision_ignore_path).with_name("ignore.capsule.yml")
            ),
        )
        if missing:
            Path(collision_config.get_self_collision_ignore_path("sphere")).unlink()
        planner = WarpPlanner(
            MotionPlannerConfig(num_steps_pre_grasp=4, traj_opt_self_collision_weight=float(enabled)), collision_config
        )
        pairs = planner._warp_robot.spec.self_collision_ignored_pairs
        expected = (
            [] if missing else [[planner._warp_robot.link_names.index("a"), planner._warp_robot.link_names.index("b")]]
        )
        np.testing.assert_array_equal(pairs, np.array(expected, dtype=np.int32).reshape(-1, 2))
        assert "self-collision ignore path" not in caplog.text
        solver = Mock()
        solver.return_value.solve_offline_numpy.return_value.q_traj = np.zeros((1, 4, 3), dtype=np.float32)
        monkeypatch.setattr("graspkit.stages.motion_plan.warp_planner.MotionPlanner", solver)
        scene = SceneGeom([SceneDescription(boxes={"box": BoxDescription("box", (0.1, 0.1, 0.1), (3.0, 0.0, 0.0))})])
        planner.optimize_trajs(torch.zeros(1, 1, 4, 3), scene)
        solver.assert_called_once()
        assert solver.call_args.args[1] is planner._warp_robot
        messages = [r.message for r in caplog.records if "self-collision ignore path" in r.message]
        assert len(messages) == int(enabled and missing)
        if messages:
            assert collision_config.name in messages[0] and "sphere" in messages[0]

    def test_joint_velocity_limits_override_the_urdf(self, collision_config: RobotConfig):
        robot = replace(collision_config, joint_velocity_limits={"b_joint": 3.0})
        planner = WarpPlanner(MotionPlannerConfig(num_steps_pre_grasp=4), robot)
        np.testing.assert_array_equal(planner._warp_robot.spec.actuated_joint_velocity_limits, [1.0, 3.0, 1.0])

    @pytest.mark.parametrize("close", [True, False])
    def test_plan_full_passes_every_phase_and_holds(self, collision_config: RobotConfig, tmp_path: Path, close: bool):
        # an arm joint a carrying the hand joints b and c
        urdf = Path(collision_config.urdf_or_mjcf_path).read_text()
        urdf = urdf.replace('<parent link="base"/><child link="b"/>', '<parent link="a"/><child link="b"/>')
        urdf = urdf.replace('<parent link="base"/><child link="c"/>', '<parent link="a"/><child link="c"/>')
        (tmp_path / "arm.urdf").write_text(urdf)
        robot = replace(collision_config, urdf_or_mjcf_path=str(tmp_path / "arm.urdf"), hand_base_link_name="a")
        cfg = MotionPlannerConfig(
            num_steps_pre_grasp=500,
            num_steps_pre_to_grasp=5,
            num_steps_holding=10,
            num_steps_post_grasp=5,
            close_during_holding=close,
            traj_opt_max_iter=5,
            enable_joint_optimization=True,
        )
        planner = WarpPlanner(cfg, robot)
        start, pre, grasp, post, lift = (
            torch.tensor(q).expand(2, 3, 3)
            for q in ([0.0, 0, 0], [0.5, 0.5, 0.5], [0.6, 0.7, 0.8], [0.6, 0.7, 0.9], [0.4, 0.7, 0.9])
        )
        # the box covers link c, so a collision check past the pre-grasp would rule the grasp out
        scene = SceneGeom(
            [SceneDescription(boxes={"box": BoxDescription("box", (0.1, 0.1, 0.1), (0.6, 0.0, 0.0))})] * 2
        )
        trajs, pene = planner.plan_full(start, pre, grasp, post, lift, scene)
        assert trajs.shape[:2] == (2, 3) and trajs.shape[2] >= cfg.num_steps and pene.shape == (2, 3)
        torch.testing.assert_close(trajs[..., 0, :], start)
        torch.testing.assert_close(trajs[..., -1, :], lift)  # padded with the lift
        at = int((trajs[0, 0] - grasp[0, 0]).abs().amax(dim=-1).lt(1e-6).nonzero()[0])
        torch.testing.assert_close(trajs[..., at + int(not close), :], post if not close else grasp)
        torch.testing.assert_close(trajs[..., at + 10, :], post)  # the hold keeps its 10 steps
        # a straight hold: the hand closes over it, or at once and then holds
        torch.testing.assert_close(trajs[..., at + 5, :], (grasp + post) / 2 if close else post)
