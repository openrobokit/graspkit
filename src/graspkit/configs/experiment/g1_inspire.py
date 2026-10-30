from dataclasses import replace

from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_3_small_table, dex1b_v0_4
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
from graspkit.configs.grasp_policy import pointnet_0_1_kl, pointnet_basic
from graspkit.configs.robot.g1_inspire import g1_inspire_right
from graspkit.constants import DATA_ROOT

g1_inspire_right_dex1b_mini_tabletop = ExperimentConfig(
    name="g1_inspire_right_dex1b_mini_tabletop",
    robot_config=g1_inspire_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right,
        floating_base_robot_config=g1_inspire_right,  # policy predicts fixed-base qpos; planned as-is
        motion_planner=MotionPlannerConfig(
            backend="warp",
            num_samples_per_batch=1024,
            post_grasp_hand_grad_move=0.5,
            num_steps_pre_grasp=10,
            num_steps_pre_to_grasp=20,
            num_steps_holding=30,
            num_steps_post_grasp=50,
            lifting_dist=0.15,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(0.3, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=128,
        num_attempts=10,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=1024,
            num_seeds=2,
            perturbation_magnitude=2.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
        ),
    ),
    agent=pointnet_0_1_kl,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=128,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            contact_offset=0.005,
            enable_perturbation=False,
        ),
    ),
    steps=(
        "render_data",
        "optimize_grasps",
        "process_sim_data",
        "train",
    ),
    scene_description_path=str(dex1b_v0_3_small_table.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_grasps/dex1b_v0_3_tabletop_mini"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right")),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.3_tabletop_mini/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_traj/dex1b_tabletop_mini.h5"),
)

g1_inspire_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="g1_inspire_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=g1_inspire_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right,
        floating_base_robot_config=g1_inspire_right,  # policy predicts fixed-base qpos; planned as-is
        motion_planner=MotionPlannerConfig(
            backend="warp",
            num_samples_per_batch=1024,
            post_grasp_hand_grad_move=0.5,
            num_steps_pre_grasp=10,
            num_steps_pre_to_grasp=20,
            num_steps_holding=30,
            num_steps_post_grasp=50,
            lifting_dist=0.15,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(0.3, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=1024,
        num_attempts=1,
        # optimizer frees right arm (7) + right hand (6), the topology default (waist3+arm7+hand6, waist ignored)
        optimizer_backend="warp_gd",
        num_samples_per_scene=512,
        num_scenes_per_batch=4,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=1024,
            num_seeds=2,
            perturbation_magnitude=2.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
        ),
    ),
    agent=pointnet_basic,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=128,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=1024,
            num_seeds=2,
            perturbation_magnitude=2.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_grasps/dex1b_v0_4_tabletop_mini"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right")),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_traj/dex1b_v0_4_tabletop_mini.h5"),
)

g1_inspire_right_dex1b_v0_4_dev = replace(
    g1_inspire_right_dex1b_v0_4_mini_tabletop,
    name="g1_inspire_right_dex1b_v0_4_dev",
    scene_description_path=str(dex1b_v0_4.TABLETOP_DEV_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_grasps/dex1b_v0_4_dev"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_dev/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_traj/dex1b_v0_4_dev.h5"),
)
