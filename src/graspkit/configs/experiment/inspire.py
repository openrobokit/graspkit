import math
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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_6drot, spconv_layernorm_local_kin_loss_6drot
from graspkit.configs.robot.g1_inspire import g1_inspire_right
from graspkit.configs.robot.inspire import inspire_right
from graspkit.constants import DATA_ROOT

inspire_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="inspire_right_dex1b_v0_4_mini_floating",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=inspire_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=inspire_right,
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            pre_grasp_hand_grad_move=1.0,
            post_grasp_hand_grad_move=0.5,
            num_steps_post_grasp=100,
            lifting_dist=0.20,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(0.3, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=128,
        num_attempts=10,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        # in-loop filter: heavier object + external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            floating_object=True,
            perturbation_magnitude=5.0,
        ),
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"), load_scenes=True, floating_object=True
        ),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=128,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/inspire_grasps/dex1b_v0_4_floating_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.3_tabletop_mini/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_traj/dex1b_tabletop_mini.h5"),
)

inspire_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="inspire_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=inspire_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=inspire_right,
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            pre_grasp_hand_grad_move=1.0,
            post_grasp_hand_grad_move=0.5,
            num_steps_post_grasp=100,
            lifting_dist=0.20,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(0.3, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=128,
        num_attempts=10,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        # in-loop filter: heavier object + external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            floating_object=False,
            perturbation_magnitude=5.0,
        ),
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"), load_scenes=True, floating_object=True
        ),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=128,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=False,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/inspire_grasps/dex1b_v0_4_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/inspire_traj/dex1b_v0_4_tabletop_mini.h5"),
)

inspire_right_dex1b_v0_4_dev = ExperimentConfig(
    name="inspire_right_dex1b_v0_4_dev",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=inspire_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=inspire_right,
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            pre_grasp_hand_grad_move=1.0,
            post_grasp_hand_grad_move=0.5,
            num_steps_post_grasp=100,
            lifting_dist=0.20,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(0.3, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=128,
        num_attempts=10,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
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
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"), load_scenes=True, floating_object=False
        ),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=128,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
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
    grasp_output_dir=str(DATA_ROOT / "optim_data/inspire_grasps/dex1b_v0_4_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/pointcloud/dex1b/inspire_dex1b_v0_4_dev.h5"),
    processed_data_path=str(DATA_ROOT / "sim_data/inspire_traj/dex1b_v0_4_dev_floating.h5"),
)

inspire_right_dex1b_v0_5_mini_tabletop = replace(
    inspire_right_dex1b_v0_4_mini_tabletop,
    name="inspire_right_dex1b_v0_5_mini_tabletop",
    enable_contact_extraction=True,
    grasp_optim=replace(
        inspire_right_dex1b_v0_4_mini_tabletop.grasp_optim,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
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
    agent=replace(spconv_layernorm_local_kin_loss_6drot, kl_weight=0.15, tl_weight=1.0),
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True, num_actions=16, return_graspness=True),
    trainer=replace(
        inspire_right_dex1b_v0_4_mini_tabletop.trainer,
        batch_size=64,
        lr=2e-4,
        total_iters=90000,
        save_freq=10000,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="inspire_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=1024,
            num_seeds=2,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=False,
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/inspire_grasps/dex1b_v0_5_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/inspire"),
    processed_data_path=str(DATA_ROOT / "sim_data/inspire_traj/dex1b_v0_5_tabletop_mini_contact.h5"),
)
