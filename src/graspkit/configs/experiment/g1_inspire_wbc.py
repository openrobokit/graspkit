from dataclasses import replace

from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_4_1, dex1b_v0_5_0
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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_kin_loss, pointnet_0_1_kl_kin_loss_pen
from graspkit.configs.robot.g1_inspire_wbc import g1_inspire_right_wbc
from graspkit.constants import DATA_ROOT, S3_ROOT

g1_inspire_right_wbc_dex1b_v0_4_1_mini = ExperimentConfig(
    name="g1_inspire_right_wbc_dex1b_v0_4_1_mini",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=g1_inspire_right_wbc,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_inspire_right_wbc,
        floating_base_robot_config=g1_inspire_right_wbc,  # policy predicts fixed-base qpos; planned as-is
        pen_filter=True,
        pen_filter_threshold=0.002,
        plan_table_only=True,
        motion_planner=MotionPlannerConfig(
            backend="warp",
            num_samples_per_batch=1024,
            post_grasp_hand_grad_move=0.25,
            pre_grasp_arm_grad_move=0.002,
            num_steps_pre_grasp=10,
            num_steps_pre_to_grasp=20,
            num_steps_holding=30,
            num_steps_post_grasp=50,
            lifting_dist=0.15,
            # height_base + waist yaw/roll/pitch: held at grasp value (first 4 joints of full robot and pruned chain).
            start_from_grasp_dof_indices=(0, 1, 2, 3),
            pen_weight=10.0,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right_wbc"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=1024,
            num_seeds=2,
            perturbation_magnitude=2.0,
            perturbation_step_range=(-25, -5),
            enable_perturbation=True,
            num_perturbations=4,
        ),
        hand_qpos_mu=(0.5, 0.05, 0.6, 0.6, 0.6, 0.6),
        min_grasps_threshold=128,
        num_attempts=3,
        num_samples_per_scene=256,
        num_scenes_per_batch=4,
        distance_lower=0.08,
        distance_upper=0.30,
        optimizer_backend="warp_gd",
        sampler_type="wbc",
    ),
    agent=pointnet_0_1_kl_kin_loss,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=256,
        total_iters=100000,
        # Trainer eval: light object, no perturbation (pre-#129 tuning; #129 misrouted it to the harsh filter above).
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right_wbc"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            contact_offset=0.005,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4_1.SMALL_WORLD_TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_wbc_grasps/dex1b_v0_4_1"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="g1_inspire_right_wbc")),
        camera_poses_path=str(calib.calib_file("g1/zedmini_720p_camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/zedmini_720p_intrinsic.json")),
        render_downscale=2.0,  # 1280x720 -> 640x360 (4x fewer pixels) for faster sampled multi-view render
        look_at_object=True,
        # sampled multi-view obs: eye jittered +/-0.2m xyz, orientation +/-2deg roll-pitch / +/-5deg yaw.
        camera_mode="sampled",
        camera_sample_translation=(0.2, 0.2, 0.2),
        camera_sample_rotation=(2.0, 2.0, 5.0),  # (roll, pitch, yaw) in the camera frame
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4.1/g1_04_30"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_wbc/dex1b_v0_4_1_g1_inspire_wbc.h5"),
)
# Same recipe as the v0.4.1 mini config, but writes grasps to s3 (for cluster runs).
g1_inspire_right_wbc_dex1b_v0_4_1_mini_s3 = replace(
    g1_inspire_right_wbc_dex1b_v0_4_1_mini,
    name="g1_inspire_right_wbc_dex1b_v0_4_1_mini_s3",
    grasp_output_dir=str(S3_ROOT / "data/optim_data/g1_inspire_wbc_grasps/dex1b_v0_4_1"),
)

# v0.5 small (larger 0.35x0.35x0.2 table): v0.4.1 mini recipe on v0.5.0 wbc_tabletop_small, larger perturbation.
g1_inspire_right_wbc_dex1b_v0_5_small = replace(
    g1_inspire_right_wbc_dex1b_v0_4_1_mini,
    name="g1_inspire_right_wbc_dex1b_v0_5_small",
    scene_description_path=str(dex1b_v0_5_0.WBC_TABLETOP_SMALL_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_wbc_grasps/dex1b_v0_5_small"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0/g1_inspire_wbc_small"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_wbc/dex1b_v0_5_small_g1_inspire_wbc.h5"),
    dataset=replace(g1_inspire_right_wbc_dex1b_v0_4_1_mini.dataset, num_actions=32),
    # small per-batch scene count keeps each s3 job light/parallelizable (batch = 2 x 512 = 1024).
    grasp_optim=replace(
        g1_inspire_right_wbc_dex1b_v0_4_1_mini.grasp_optim,
        evaluator=replace(  # slightly larger perturbation on the in-loop filter
            g1_inspire_right_wbc_dex1b_v0_4_1_mini.grasp_optim.evaluator,
            perturbation_magnitude=2.0,
            perturbation_duration=2,
        ),
        min_grasps_threshold=32,
        num_samples_per_scene=512,
        num_scenes_per_batch=2,
        num_attempts=5,
    ),
)

# Same recipe as v0_5_small, but writes grasps to s3 (for work-stealing cluster workers).
g1_inspire_right_wbc_dex1b_v0_5_small_s3 = replace(
    g1_inspire_right_wbc_dex1b_v0_5_small,
    name="g1_inspire_right_wbc_dex1b_v0_5_small_s3",
    grasp_output_dir=str(S3_ROOT / "data/optim_data/g1_inspire_wbc_grasps/dex1b_v0_5_small"),
)

# v0.5 mini: same recipe as v0_5_small, but points at the smaller wbc_tabletop_mini scene set.
g1_inspire_right_wbc_dex1b_v0_5_mini = replace(
    g1_inspire_right_wbc_dex1b_v0_5_small,
    name="g1_inspire_right_wbc_dex1b_v0_5_mini",
    scene_description_path=str(dex1b_v0_5_0.WBC_TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_wbc_grasps/dex1b_v0_5_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0/g1_inspire_wbc_mini"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_inspire_wbc/dex1b_v0_5_mini_g1_inspire_wbc.h5"),
)

# Same recipe as v0_5_mini, but writes grasps to s3 (for work-stealing cluster workers).
g1_inspire_right_wbc_dex1b_v0_5_mini_s3 = replace(
    g1_inspire_right_wbc_dex1b_v0_5_mini,
    name="g1_inspire_right_wbc_dex1b_v0_5_mini_s3",
    grasp_output_dir=str(S3_ROOT / "data/optim_data/g1_inspire_wbc_grasps/dex1b_v0_5_mini"),
)

# Penetration-loss variant of v0_5_mini, with a short warmup so the pen regularizer kicks in early.
g1_inspire_right_wbc_dex1b_v0_5_mini_pen = replace(
    g1_inspire_right_wbc_dex1b_v0_5_mini,
    name="g1_inspire_right_wbc_dex1b_v0_5_mini_pen",
    agent=replace(pointnet_0_1_kl_kin_loss_pen, penetration_warmup_iters=5000, penetration_every=10),
    trainer=replace(g1_inspire_right_wbc_dex1b_v0_5_mini.trainer, total_iters=120000),
)

# Top-down pen variant: 55deg cone (shared by init sampling and the post-eval filter) at the success-median angle.
g1_inspire_right_wbc_dex1b_v0_5_mini_pen_samefilter = replace(
    g1_inspire_right_wbc_dex1b_v0_5_mini_pen,
    name="g1_inspire_right_wbc_dex1b_v0_5_mini_pen_samefilter",
    grasp_optim=replace(
        g1_inspire_right_wbc_dex1b_v0_5_mini.grasp_optim,
        hand_axis="palm_right",
        hand_axis_target=(0.0, 0.0, -1.0),
        filter_hand_axis_max_angle_deg=55.0,
        num_scenes_per_batch=8,
        num_attempts=1,
        max_scenes=24,
    ),
    traj_builder=replace(
        g1_inspire_right_wbc_dex1b_v0_5_mini_pen.traj_builder,
        motion_planner=replace(
            g1_inspire_right_wbc_dex1b_v0_5_mini_pen.traj_builder.motion_planner, post_grasp_hand_grad_move=1.0
        ),
        hand_axis="palm_right",
        hand_axis_target=(0.0, 0.0, -1.0),
        filter_hand_axis_max_angle_deg=55.0,
    ),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_inspire_wbc_grasps/dex1b_v0_5_mini_pen_samefilter_traj_builder"),
)
