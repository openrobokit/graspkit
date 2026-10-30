from dataclasses import replace

from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_4
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
from graspkit.configs.grasp_policy import spconv_layernorm_local_kin_loss
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot.xarm7_ability import xarm7_ability_right_centered
from graspkit.constants import DATA_ROOT

xarm7_ability_right_centered_pointnet = ExperimentConfig(
    name="xarm7_ability_right_centered_pointnet",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=xarm7_ability_right_centered,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_ability_right_centered,
        floating_base_robot_config=xarm7_ability_right_centered,  # policy predicts fixed-base qpos; planned as-is
        motion_planner=MotionPlannerConfig(
            num_steps_holding=25,
            num_samples_per_batch=1024,
            opt_steps=300,
        ),
    ),
    agent=GraspPolicyConfig(vision_encoder="pointnet", kl_weight=0.01),
    dataset=DatasetConfig(num_points=2048, num_actions=32),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(-1.5, 0.0, 0.27, 0.27, 0.27, 0.27),
        spen_threshold=0.0,  # see configs/experiment/ability.py
        optimizer_backend="warp_gd",
        num_samples_per_scene=256,
        num_scenes_per_batch=8,
        # in-loop filter: external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xarm7_ability_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.2,
            num_envs=512,
            enable_perturbation=True,
            num_perturbations=4,
            perturbation_step_range=(195, 240),
            perturbation_duration=3,
            perturbation_magnitude=30,
        ),
    ),
    trainer=TrainerConfig(
        batch_size=64,
        total_iters=100_000,
        eval_freq=10_000,
        save_freq=10_000,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xarm7_ability_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.2,
            num_envs=512,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    enable_contact_extraction=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xarm_ability_grasps/dex1b_v0_4_tabletop_mini_center_pose"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="xarm7_ability_right")),
        camera_poses_path=str(calib.calib_file("xarm/26-05-16-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/xarm_ability_center_pose_26-05-16"),
    processed_data_path=str(DATA_ROOT / "sim_data/ability_grasps/xarm_ability_center_pose_dex1b_v0_4_tabletop_mini.h5"),
)

xarm7_ability_right_centered_spconv_layernorm_local_01kl = replace(
    xarm7_ability_right_centered_pointnet,
    name="xarm7_ability_right_centered_spconv_layernorm_local_01kl",
    agent=spconv_layernorm_local_kin_loss,
    dataset=DatasetConfig(
        num_points=2048,
        num_actions=32,
        return_graspness=True,
        return_object_pointcloud_only=True,
    ),
)
