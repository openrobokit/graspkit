"""Fixed-base xArm7 + xHand grasp-opt; unlike Shadow, the fully-actuated xHand's parse_mjcf FK matches MuJoCo.

`MjEnv` (`grasp_optim.evaluator.mj_env_config`, tabletop) in-loop-filters grasps:
- grasp the resting object, apply a downward wrench, lift the palm 10cm via arm IK
- keep grasps whose object follows
"""

import os

from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.mj_hand import XARM_XHAND
from graspkit.configs.robot.xarm7_xhand_mj import xarm7_xhand_right_mj
from graspkit.constants import DATA_ROOT

xarm7_xhand_right_mj_tabletop = ExperimentConfig(
    name="xarm7_xhand_right_mj_tabletop",
    steps=("optimize_grasps",),
    robot_config=xarm7_xhand_right_mj,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_xhand_right_mj,
        floating_base_robot_config=xarm7_xhand_right_mj,  # optimized fixed-base grasps planned as-is
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(post_grasp_hand_grad_move=0.1, num_steps_pre_to_grasp=40),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(setting="tabletop", hand=XARM_XHAND, fc_force_scale=1.0, timestep=0.001)
        ),  # in-loop grasp filter
        num_samples_per_scene=64,
        num_scenes_per_batch=10,
        min_grasps_threshold=16,
        num_attempts=1,
        # Match the working xarm7_xhand_right recipe (GD-based, palm allowed close to the object).
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.025,
        optimizer_backend="warp_gd",
    ),
    # timestep=0.001: actuated arm+hand; 0.004 flings the object & fingers never settle.
    scene_description_path=os.path.join(DATA_ROOT, "scene_description/shadow_mj/dexgraspnet_tabletop_subset.msgpack"),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/xarm_xhand_mj/dexgraspnet_tabletop_subset"),
)
