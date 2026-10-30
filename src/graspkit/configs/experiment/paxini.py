from graspkit.assets.scene_descriptions import dex1b_v0_4
from graspkit.configs.experiment_config import (
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.robot.paxini import paxini_right
from graspkit.constants import DATA_ROOT

paxini_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="paxini_right_dex1b_v0_4_mini_floating",
    steps=("optimize_grasps",),
    robot_config=paxini_right,
    traj_builder=GraspTrajBuilderConfig(
        floating_base_robot_config=paxini_right,
        motion_planner=MotionPlannerConfig(pre_grasp_hand_grad_move=1.0),
    ),
    grasp_optim=GraspOptimConfig(
        spen_threshold=0.0,  # the 1 cm margin held fingertips off the object (>=3 within 3 mm: 73 -> 79%)
    ),
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/paxini_grasps/dex1b_v0_4_floating_mini"),
)
