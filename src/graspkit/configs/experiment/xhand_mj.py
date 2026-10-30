"""Floating xHand grasp-opt -> in-loop MuJoCo lift validation; no ManiSkill agent, so MjEnv is the in-loop filter."""

import math
import os

from graspkit.assets.scene_descriptions import dex1b_v0_5_0
from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.mj_hand import XHAND
from graspkit.configs.robot.xhand_mj import xhand_right_mj
from graspkit.constants import DATA_ROOT

xhand_right_mj_dex1b_v0_5_tabletop = ExperimentConfig(
    name="xhand_right_mj_dex1b_v0_5_tabletop",
    steps=("optimize_grasps",),
    robot_config=xhand_right_mj,
    traj_builder=GraspTrajBuilderConfig(
        floating_base_robot_config=xhand_right_mj,
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(pre_grasp_hand_grad_move=0.2, post_grasp_hand_grad_move=0.1),
    ),
    grasp_optim=GraspOptimConfig(
        # timestep=0.001 keeps the actuated fingers stable during the tabletop grasp-then-lift-against-force test.
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(setting="tabletop", hand=XHAND, fc_force_scale=1.0, timestep=0.001)
        ),
        num_samples_per_scene=64,
        num_scenes_per_batch=10,
        min_grasps_threshold=16,
        num_attempts=1,
        # Same 12-joint mjcf-order init the xarm7_xhand_right_mj recipe uses.
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.025,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        w_angle_z=100.0,
        optimizer_backend="warp_gd",
        sampler_type="floating_base",
    ),
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/xhand_mj/dex1b_v0_5_tabletop_mini"),
)
