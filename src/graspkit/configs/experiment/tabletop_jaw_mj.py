"""RoboDojo ARX X5 and RoboTwin AgileX ALOHA on the Dex1B tabletop scenes."""

from dataclasses import replace

from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.robot.agilex_aloha_mj import AGILEX_ALOHA, agilex_aloha_arm_mj
from graspkit.configs.robot.arx_x5_mj import ARX_X5, arx_x5_arm_mj
from graspkit.constants import DATA_ROOT

arx_x5_arm_mj_tabletop = ExperimentConfig(
    name="arx_x5_arm_mj_tabletop",
    steps=("optimize_grasps",),
    robot_config=arx_x5_arm_mj,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=arx_x5_arm_mj,
        floating_base_robot_config=arx_x5_arm_mj,
        motion_planner=MotionPlannerConfig(pre_grasp_arm_grad_move=0.005),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(
                setting="tabletop",
                hand=ARX_X5,
                fc_force_scale=1.0,
                timestep=0.001,
                num_workers=8,
                angle_thre=360.0,
                min_lift_dist=0.1,
            )
        ),
        num_samples_per_scene=32,
        num_scenes_per_batch=1,
        min_grasps_threshold=1,
        num_attempts=1,
        optimizer_backend="warp_gd",
    ),
    scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini.msgpack"),
    grasp_output_dir=str(DATA_ROOT / "optim_data/arx_x5_mj/dex1b_v0_5_tabletop_mini"),
)

agilex_aloha_arm_mj_tabletop = replace(
    arx_x5_arm_mj_tabletop,
    name="agilex_aloha_arm_mj_tabletop",
    robot_config=agilex_aloha_arm_mj,
    traj_builder=replace(
        arx_x5_arm_mj_tabletop.traj_builder,
        fixed_base_robot_config=agilex_aloha_arm_mj,
        floating_base_robot_config=agilex_aloha_arm_mj,
    ),
    grasp_optim=replace(
        arx_x5_arm_mj_tabletop.grasp_optim,
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(
                setting="tabletop",
                hand=AGILEX_ALOHA,
                fc_force_scale=1.0,
                timestep=0.001,
                num_workers=8,
                angle_thre=360.0,
                min_lift_dist=0.1,
            )
        ),
    ),
    grasp_output_dir=str(DATA_ROOT / "optim_data/agilex_aloha_mj/dex1b_v0_5_tabletop_mini"),
)
