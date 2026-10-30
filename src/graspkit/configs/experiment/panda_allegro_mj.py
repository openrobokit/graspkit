"""DexJoCo Panda and Allegro on the Dex1B tabletop scenes."""

from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.robot.panda_allegro_mj import PANDA_ALLEGRO, panda_allegro_right_mj
from graspkit.constants import DATA_ROOT

panda_allegro_right_mj_tabletop = ExperimentConfig(
    name="panda_allegro_right_mj_tabletop",
    steps=("optimize_grasps",),
    robot_config=panda_allegro_right_mj,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=panda_allegro_right_mj,
        floating_base_robot_config=panda_allegro_right_mj,
        motion_planner=MotionPlannerConfig(pre_grasp_arm_grad_move=0.005, post_grasp_hand_grad_move=1.0),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            mj_env_config=MjEvalConfig(
                setting="tabletop",
                hand=PANDA_ALLEGRO,
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
    grasp_output_dir=str(DATA_ROOT / "optim_data/panda_allegro_mj/dex1b_v0_5_tabletop_mini"),
)
