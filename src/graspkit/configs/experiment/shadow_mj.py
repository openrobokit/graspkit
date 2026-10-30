"""Shadow-hand grasp-opt -> MuJoCo validation on a DexGraspBench mjcf (no ManiSkill agent, `MjEnv` filters in-loop)."""

import dataclasses
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
from graspkit.configs.robot.shadow_mj import shadow_right_mj
from graspkit.constants import DATA_ROOT

# Curled init (rh_ FF/MF/RF/LF/TH order, thumb opposition THJ4) so sampled grasps start wrapping, not flat-open.
_SHADOW_INIT = (
    0.0,
    0.3,
    0.4,
    0.4,  # FF: J4 spread, J3/J2/J1 flex
    0.0,
    0.3,
    0.4,
    0.4,  # MF
    0.0,
    0.3,
    0.4,
    0.4,  # RF
    0.0,
    0.0,
    0.3,
    0.4,
    0.4,  # LF: J5 metacarpal, J4 spread, J3/J2/J1 flex
    0.2,
    0.8,
    0.2,
    0.3,
    0.0,  # TH: J5 base, J4 opposition, J3/J2/J1
)

shadow_right_mj_val = ExperimentConfig(
    name="shadow_right_mj_val",
    steps=("optimize_grasps",),
    robot_config=shadow_right_mj,
    traj_builder=GraspTrajBuilderConfig(
        floating_base_robot_config=shadow_right_mj,
        motion_planner=MotionPlannerConfig(pre_grasp_hand_grad_move=1.0),
    ),
    grasp_optim=GraspOptimConfig(
        # No ManiSkill agent for menagerie shadow; MjEnv (mj_env_config) is the in-loop grasp filter.
        evaluator=EnvConfig(mj_env_config=MjEvalConfig()),
        num_samples_per_scene=16,
        num_scenes_per_batch=10,
        min_grasps_threshold=16,
        num_attempts=1,
        hand_qpos_mu=_SHADOW_INIT,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
    ),
    scene_description_path=os.path.join(DATA_ROOT, "scene_description/shadow_mj/dexgraspnet_subset.msgpack"),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/shadow_mj/dexgraspnet_subset"),
)

# Tabletop variant: object rests on a table box; MjEnv runs the lift test (grasp+lift 10cm) instead of force-closure.
shadow_right_mj_val_tabletop = dataclasses.replace(
    shadow_right_mj_val,
    name="shadow_right_mj_val_tabletop",
    grasp_optim=dataclasses.replace(
        shadow_right_mj_val.grasp_optim, evaluator=EnvConfig(mj_env_config=MjEvalConfig(setting="tabletop"))
    ),
    scene_description_path=os.path.join(DATA_ROOT, "scene_description/shadow_mj/dexgraspnet_tabletop_subset.msgpack"),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/shadow_mj/dexgraspnet_tabletop_subset"),
)
