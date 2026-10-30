"""Fixed-base xArm7 + menagerie Shadow grasp-opt on the full arm, filtered in-loop by a MuJoCo tabletop lift check."""

import math
import os

from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.mj_hand import XARM_SHADOW
from graspkit.configs.robot.xarm7_shadow import _SHADOW_INIT, xarm7_shadow_right
from graspkit.constants import DATA_ROOT

xarm7_shadow_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="xarm7_shadow_right_dex1b_v0_4_mini_tabletop",
    steps=("optimize_grasps",),
    robot_config=xarm7_shadow_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_shadow_right,
        floating_base_robot_config=xarm7_shadow_right,  # optimized fixed-base grasps planned as-is
        pen_filter=True,
        pen_filter_threshold=0.002,
        motion_planner=MotionPlannerConfig(backend="warp"),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(setting="tabletop", hand=XARM_SHADOW, fc_force_scale=1.0)
        ),  # in-loop grasp filter
        num_samples_per_scene=64,
        num_scenes_per_batch=10,
        min_grasps_threshold=16,
        num_attempts=1,
        hand_qpos_mu=_SHADOW_INIT,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        # grasp-quality knobs mirrored from vega_sharpa_right_mj: fingertip-distance energy pulls tips onto the object
        distance_lower=0.08,
        distance_upper=0.15,
        w_spen=20.0,
        spen_threshold=0.0,
        spen_representation="sphere",
        max_scenes=8,
    ),  # in-loop MjEnv filter on the SAME combined mjcf, gentle wrench
    # dexgraspnet subset with per-convex-piece meshes on disk, resting on a table; same subset shadow_right_mj uses
    scene_description_path=os.path.join(DATA_ROOT, "scene_description/shadow_mj/dexgraspnet_tabletop_subset.msgpack"),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/xarm_shadow/dexgraspnet_tabletop_subset"),
)
