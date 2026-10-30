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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_6drot
from graspkit.configs.robot.sharpa import sharpa_left, sharpa_right
from graspkit.configs.robot.vega_sharpa import vega_sharpa_left, vega_sharpa_right
from graspkit.configs.robot.xarm6_sharpa import xarm6_sharpa_right
from graspkit.constants import DATA_ROOT

# warp_gd gives sharpa_right 0% sim success (optimizer-backend-specific bug); warp_lm gets ~5-11%.
sharpa_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="sharpa_right_dex1b_v0_4_mini_floating",
    steps=("optimize_grasps",),
    robot_config=sharpa_right,
    traj_builder=GraspTrajBuilderConfig(
        floating_base_robot_config=sharpa_right,
        motion_planner=MotionPlannerConfig(pre_grasp_hand_grad_move=1.0),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            num_envs=64,
            enable_perturbation=False,
        ),  # in-loop grasp filter
        optimizer_backend="warp_gd",
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        num_samples_per_scene=128,
        max_scenes=32,
        hand_qpos_mu=(1.0, 0.0, 0.3, 0.0, 0.3) + (0.4, 0.0, 0.4, 0.2) * 3 + (0.1, 0.4, 0.0, 0.4, 0.2),
        spen_threshold=0.0,
    ),
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/sharpa_grasps/dex1b_v0_4_floating_mini"),
)


sharpa_right_dex1b_v0_5_mini_tabletop_gd = ExperimentConfig(
    name="sharpa_right_dex1b_v0_5_mini_tabletop_gd",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=sharpa_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm6_sharpa_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=sharpa_right,
        motion_planner=MotionPlannerConfig(pre_grasp_hand_grad_move=1.0),
    ),
    grasp_optim=GraspOptimConfig(
        # in-loop grasp filter, aligned with the trainer evaluator: object on table + 8cm lift
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            num_envs=1024,
            enable_perturbation=False,
        ),
        optimizer_backend="warp_gd",
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        num_samples_per_scene=128,
        hand_qpos_mu=(1.0, 0.0, 0.3, 0.0, 0.3) + (0.4, 0.0, 0.4, 0.2) * 3 + (0.1, 0.4, 0.0, 0.4, 0.2),
        spen_threshold=0.0,
        spen_representation="sphere",
        min_grasps_threshold=1,
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"), load_scenes=True, floating_object=False
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-05-13-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=16,
        total_iters=40000,
        eval_freq=10000,
        save_freq=10000,
        # Must match grasp_optim.evaluator above, or the policy is scored under physics its data was never filtered by.
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            num_envs=1024,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/sharpa_grasps/dex1b_v0_5_mini_tabletop_gd"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/sharpa"),
    processed_data_path=str(DATA_ROOT / "sim_data/sharpa_floating/dex1b_v0_5_tabletop_mini_gd.h5"),
)

sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed = replace(
    sharpa_right_dex1b_v0_5_mini_tabletop_gd,
    name="sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed",
    # Anchor the camera to the torso it is mounted on; fixed_base_robot_config only feeds render_data.py's camera_frame.
    traj_builder=replace(
        sharpa_right_dex1b_v0_5_mini_tabletop_gd.traj_builder, fixed_base_robot_config=vega_sharpa_right
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"),
            load_scenes=True,
            floating_object=False,
            camera_near=0.05,
        ),
        camera_poses_path=str(calib.calib_file("vega_sharpa/26-08-21-finger-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("vega_sharpa/26-08-05-camera_intrinsic.json")),
    ),
    grasp_output_dir=str(DATA_ROOT / "optim_data/sharpa_grasps/dex1b_v0_5_mini_tabletop_gd_zed"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/sharpa_zed"),
    processed_data_path=str(DATA_ROOT / "sim_data/sharpa_floating/dex1b_v0_5_tabletop_mini_gd_zed.h5"),
)

sharpa_left_dex1b_v0_5_mini_tabletop_gd_zed = replace(
    sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed,
    name="sharpa_left_dex1b_v0_5_mini_tabletop_gd_zed",
    robot_config=sharpa_left,
    traj_builder=replace(
        sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.traj_builder,
        fixed_base_robot_config=vega_sharpa_left,
        floating_base_robot_config=sharpa_left,
    ),
    grasp_optim=replace(
        sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.grasp_optim,
        evaluator=replace(
            sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.grasp_optim.evaluator,
            ms_env_config=replace(
                sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.grasp_optim.evaluator.ms_env_config,
                robot_uid="sharpa_left_floating_base",
            ),
        ),
    ),
    renderer=replace(
        sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.renderer,
        env_config=replace(
            sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.renderer.env_config,
            ms_env_config=MsEnvConfig(robot_uid="sharpa_left_floating_base"),
        ),
    ),
    trainer=replace(
        sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.trainer,
        evaluator=replace(
            sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.trainer.evaluator,
            ms_env_config=replace(
                sharpa_right_dex1b_v0_5_mini_tabletop_gd_zed.trainer.evaluator.ms_env_config,
                robot_uid="sharpa_left_floating_base",
            ),
        ),
    ),
    grasp_output_dir=str(DATA_ROOT / "optim_data/sharpa_left_grasps/dex1b_v0_5_mini_tabletop_gd_zed"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/sharpa_left_zed"),
    processed_data_path=str(DATA_ROOT / "sim_data/sharpa_left_floating/dex1b_v0_5_tabletop_mini_gd_zed.h5"),
)
