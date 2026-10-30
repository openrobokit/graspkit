import math
from dataclasses import replace

from robokit.assets.robots.end_effectors import arx_x5_gripper

from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_5_0
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
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot.arx_x5_gripper_robodojo import arx_x5_gripper_robodojo
from graspkit.configs.robot.xarm7_xhand import xarm7_xhand_right
from graspkit.constants import DATA_ROOT

# The in-loop filter and the trainer's evaluator: object on the table, 100 g, closed jaws lift it 8 cm.
ARX_X5_EVAL = EnvConfig(
    load_scenes=True,
    control_freq=50,
    object_mass=0.1,
    target_lift_dist=0.08,
    num_envs=1024,
    enable_perturbation=False,
)

# RoboDojo's ARX X5 jaw on the v0.5 tabletop set: floating-base optimizer, xhand v0.5's calibrated front camera.
arx_x5_gripper_dex1b_mini_tabletop = ExperimentConfig(
    name="arx_x5_gripper_dex1b_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=arx_x5_gripper_robodojo,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_xhand_right,  # only anchors the camera (camera_frame="robot")
        floating_base_robot_config=arx_x5_gripper_robodojo,
        # teleport the open jaws to the grasp, close fully (a jaw needs the full squeeze), lift
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            num_steps_post_grasp=100,
            pre_grasp_hand_grad_move=1.0,
            post_grasp_hand_grad_move=1.0,
            lifting_dist=0.20,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=ARX_X5_EVAL,
        optimizer_backend="warp_gd",
        sampler_type="floating_base",
        hand_qpos_mu=(0.044,),  # open
        distance_lower=0.10,  # link6 sits 14.5 cm behind the pads' center
        distance_upper=0.18,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        hand_axis="palm_forward",  # approach from above or the side, never from below
        hand_axis_target=(0.0, 0.0, -1.0),
        filter_hand_axis_max_angle_deg=95.0,
        spen_threshold=0.0,
        num_samples_per_scene=128,
        min_grasps_threshold=1,
        save_labels=True,  # every grasp's ManiSkill verdicts, for comparing against Isaac Sim labels
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            load_scenes=True,
            floating_object=False,
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-05-13-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    # Like sharpa_left_plain: centering, scaling and yaw augmentation let a policy transfer to new views and positions.
    agent=GraspPolicyConfig(
        vision_encoder="pointnet",
        convert_rxyz_to_rot6d=True,
        analytic_euler=True,
        center_pointcloud=True,
        pointcloud_scale=10.0,
        train_yaw_augmentation=True,
        cvae_latent_dim=32,
        kl_weight=0.5,
        tl_weight=20.0,
        enable_kinematic_loss=True,
        use_mean=False,
    ),
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=16,
        total_iters=100000,
        eval_freq=20000,
        save_freq=20000,
        evaluator=ARX_X5_EVAL,
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/arx_x5_gripper_grasps/dex1b_v0_5_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/arx_x5_gripper"),
    processed_data_path=str(DATA_ROOT / "sim_data/arx_x5_gripper_floating/dex1b_v0_5_tabletop_mini.h5"),
)

# Diverse grasps on RoboDojo-like objects and clutter with Isaac labels.
arx_x5_diverse = replace(
    arx_x5_gripper_dex1b_mini_tabletop,
    name="arx_x5_diverse",
    steps=("render_data", "process_sim_data", "train"),
    renderer=replace(
        arx_x5_gripper_dex1b_mini_tabletop.renderer,
        camera_poses_path=str(arx_x5_gripper.HEAD_CAMERA_POSE_PATH),
        camera_intrinsic_path=str(arx_x5_gripper.HEAD_CAMERA_INTRINSIC_PATH),
        camera_frame="world",
        camera_mode="head_wrist",
        camera_sample_num_views=4,
        camera_sample_translation=(0.05, 0.05, 0.05),
        camera_sample_rotation=(3.0, 3.0, 3.0),
        extract_segments=[],
        remove_segments=["table", "robot"],
        points_crop_bounds=((-0.72, 0.72), (-0.6, 0.6), (-0.01, 0.6)),
        num_points=8192,
        num_envs=32,
    ),
    # qpos_weight 50: joint7 lives in [0, 0.044] m and weight 1 never learnt it.
    agent=replace(arx_x5_gripper_dex1b_mini_tabletop.agent, qpos_weight=50.0, kl_weight=0.1),
    dataset=DatasetConfig(
        num_points=1024,
        return_object_pointcloud_only=True,
        fuse_views=2,
        view_noise=0.003,
        point_noise=0.0005,
        aug_prob=0.85,
        aug_keep_min=0.1,
        aug_voxel_max=0.003,
        aug_drop_wrist_prob=0.25,
        aug_rgb_prob=0.3,
    ),
    # Isaac-labelled data: no ManiSkill evaluator in the loop
    trainer=replace(
        arx_x5_gripper_dex1b_mini_tabletop.trainer,
        batch_size=64,
        evaluator=None,
        total_iters=60000,
        save_freq=10000,
        eval_freq=10000,
        num_dataloader_workers=16,
    ),
    scene_description_path="runs/arx_diverse/scenes/scenes.msgpack",
    grasp_output_dir="runs/arx_diverse/grasps_isaac",
    rendered_data_path="runs/arx_diverse/render",
    processed_data_path="runs/arx_diverse/sim_data.h5",
)

# Unconditioned generators: pre on all checked grasps, ft from pre on mix1 (Isaac positives if labelled, else checked).
arx_x5_diverse_pre = replace(
    arx_x5_diverse,
    name="arx_x5_diverse_pre",
    trainer=replace(arx_x5_diverse.trainer, total_iters=24000, save_freq=4000),
    grasp_output_dir="runs/arx_diverse/grasps",
    processed_data_path="runs/arx_diverse/sim_data_pre.h5",
)
arx_x5_diverse_ft = replace(
    arx_x5_diverse,
    name="arx_x5_diverse_ft",
    trainer=replace(arx_x5_diverse.trainer, total_iters=15000, save_freq=5000),
    ckpt_path="logs/train/arx_x5_diverse_pre/checkpoints/iter_24000.pt",
    grasp_output_dir="runs/arx_diverse/grasps_mix1",
    processed_data_path="runs/arx_diverse/sim_data_mix1.h5",
)
