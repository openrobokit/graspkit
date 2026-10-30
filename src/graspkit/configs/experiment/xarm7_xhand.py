from dataclasses import replace

from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_4, dex1b_v0_5_0
from graspkit.configs.env_config import MsEnvConfig
from graspkit.configs.experiment_config import (
    DatasetConfig,
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
    RendererConfig,
    TrainerConfig,
)
from graspkit.configs.grasp_policy import pointnet_basic
from graspkit.configs.robot.xarm7_xhand import xarm7_xhand_right, xarm7_xhand_right_centered
from graspkit.constants import DATA_ROOT

# Fixed-base xarm + xhand_right: the grasp optimizer runs on the full arm (`optimize_grasps`).
xarm7_xhand_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="xarm7_xhand_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=xarm7_xhand_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_xhand_right,
        floating_base_robot_config=xarm7_xhand_right,  # policy predicts fixed-base qpos; planned as-is
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(post_grasp_hand_grad_move=0.1, num_steps_pre_to_grasp=40),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.025,
        optimizer_backend="warp_gd",
        # in-loop filter: external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xarm7_xhand_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=1.0,
            target_lift_dist=0.08,
            num_envs=1024,
            perturbation_magnitude=5.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
        ),
    ),
    agent=pointnet_basic,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=16,
        total_iters=40000,
        eval_freq=10000,
        save_freq=10000,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xarm7_xhand_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=1.0,
            target_lift_dist=0.08,
            num_envs=1024,
            perturbation_magnitude=5.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xarm_xhand_grasps/dex1b_v0_4_tabletop_mini"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="xarm7_xhand_right")),
        camera_poses_path=str(calib.calib_file("xarm/26-04-27-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/xarm_26-04-27"),
    processed_data_path=str(DATA_ROOT / "sim_data/xarm_xhand_traj/dex1b_v0_4_tabletop_mini.h5"),
)

# v0.5 tabletop: v0.5.0 `tabletop` set (larger 0.5x0.5x0.2); overrides the dead v0.4 camera path with the 26-05-13 rig.
xarm7_xhand_right_dex1b_v0_5_mini_tabletop = replace(
    xarm7_xhand_right_dex1b_v0_4_mini_tabletop,
    name="xarm7_xhand_right_dex1b_v0_5_mini_tabletop",
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xarm_xhand_grasps/dex1b_v0_5_tabletop_mini"),
    renderer=replace(
        xarm7_xhand_right_dex1b_v0_4_mini_tabletop.renderer,
        camera_poses_path=str(calib.calib_file("xarm/26-05-13-camera_pose.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/xarm_26-05-13"),
    processed_data_path=str(DATA_ROOT / "sim_data/xarm_xhand_traj/dex1b_v0_5_tabletop_mini.h5"),
)

# Same v0.5 recipe but the arm base centered (y=0) like the mjcf run, via xarm7_xhand_right_centered.
xarm7_xhand_right_dex1b_v0_5_mini_tabletop_centered = replace(
    xarm7_xhand_right_dex1b_v0_5_mini_tabletop,
    name="xarm7_xhand_right_dex1b_v0_5_mini_tabletop_centered",
    robot_config=xarm7_xhand_right_centered,
    traj_builder=replace(
        xarm7_xhand_right_dex1b_v0_4_mini_tabletop.traj_builder,
        fixed_base_robot_config=xarm7_xhand_right_centered,
        floating_base_robot_config=xarm7_xhand_right_centered,
    ),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xarm_xhand_grasps/dex1b_v0_5_tabletop_mini_centered"),
)
