import dataclasses
from typing import Optional

from torch.types import Device

from graspkit.configs.experiment_config import ExperimentConfig, MotionPlannerConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.stages.grasp_traj_build.builder import GraspTrajBuilder
from graspkit.stages.grasp_traj_build.ik_convert import ik_convert_floating_to_fixed, ik_convert_floating_to_fixed_batch


def build_planning_traj_builder(
    cfg: ExperimentConfig,
    device: Device,
    motion_planner: Optional[MotionPlannerConfig] = None,
    robot_config: Optional[RobotConfig] = None,
) -> GraspTrajBuilder:
    """Build a GraspTrajBuilder for `robot_config` as-is (no floating->fixed conversion), inheriting traj_builder."""
    robot_config = robot_config if robot_config is not None else cfg.robot_config
    return GraspTrajBuilder(
        dataclasses.replace(
            cfg.traj_builder,
            fixed_base_robot_config=robot_config,
            floating_base_robot_config=robot_config,
            convert_to_fixed=False,
            motion_planner=motion_planner if motion_planner is not None else cfg.traj_builder.motion_planner,
        ),
        device,
    )


__all__ = [
    "GraspTrajBuilder",
    "build_planning_traj_builder",
    "ik_convert_floating_to_fixed",
    "ik_convert_floating_to_fixed_batch",
]
