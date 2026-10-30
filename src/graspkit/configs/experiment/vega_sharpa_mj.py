"""Vega+Sharpa fixed-base grasp-opt validated in-loop by MuJoCo tabletop-lift, not ManiSkill; combined mjcf robot."""

import os
from dataclasses import replace

from graspkit.assets.scene_descriptions import dex1b_v0_5_0
from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.mj_hand import VEGA_SHARPA_LEFT
from graspkit.configs.robot.vega_sharpa_mj import vega_sharpa_left_mj, vega_sharpa_right_mj
from graspkit.constants import DATA_ROOT

# timestep=0.001 avoids QACC NaN from stiff servos at the default 0.004; lift params scaled for LIFT_DIST=0.2.
_MJ_EVAL = MjEvalConfig(
    setting="tabletop",
    timestep=0.001,
    fc_force_scale=1.0,
    miu_coef=(1.0, 0.02),
    angle_thre=30.0,
    grasp_interp_steps=25,
    lift_interp_steps=120,
    lift_slip_thre=0.04,
)

vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop = ExperimentConfig(
    name="vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop",
    steps=("optimize_grasps",),
    robot_config=vega_sharpa_right_mj,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=vega_sharpa_right_mj,
        floating_base_robot_config=vega_sharpa_right_mj,  # optimized fixed-base grasps planned as-is
        pen_filter=True,
        pen_filter_threshold=0.002,
        # Plan against the table only: home pose hovers over the object, so start-frame overlap would fail every mp.
        plan_table_only=True,
        motion_planner=MotionPlannerConfig(
            backend="warp",
            post_grasp_hand_grad_move=0.3,
            close_during_holding=True,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(mj_env_config=_MJ_EVAL),  # in-loop grasp filter
        sampler_type="wbc",  # humanoid-over-table; frees right arm + right hand (active indices None)
        distance_lower=0.08,
        distance_upper=0.30,
        w_spen=20.0,
        spen_threshold=0.0,
        spen_representation="sphere",
        num_samples_per_scene=64,
        max_scenes=16,
    ),
    # MjEnv (mj_env_config) judges in-loop (MuJoCo tabletop lift, no ManiSkill agent) instead of MsEnv.
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/vega_sharpa_mj/dex1b_v0_5_mini"),
)


# Left-hand mirror: same knobs; MjEnv must be told to lift/score the LEFT hand (default HandSpec is the right one).
vega_sharpa_left_mj_dex1b_v0_5_mini_tabletop = replace(
    vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop,
    name="vega_sharpa_left_mj_dex1b_v0_5_mini_tabletop",
    robot_config=vega_sharpa_left_mj,
    traj_builder=replace(
        vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop.traj_builder,
        fixed_base_robot_config=vega_sharpa_left_mj,
        floating_base_robot_config=vega_sharpa_left_mj,
    ),
    grasp_optim=replace(
        vega_sharpa_right_mj_dex1b_v0_5_mini_tabletop.grasp_optim,
        evaluator=EnvConfig(mj_env_config=replace(_MJ_EVAL, hand=VEGA_SHARPA_LEFT)),
    ),
    grasp_output_dir=os.path.join(DATA_ROOT, "optim_data/vega_sharpa_mj/dex1b_v0_5_mini_left"),
)
