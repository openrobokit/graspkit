import math
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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_6drot
from graspkit.configs.robot.ability import ability_right
from graspkit.configs.robot.xarm7_ability import xarm7_ability_right
from graspkit.constants import DATA_ROOT

_ABILITY_GRASP_OPTIM = GraspOptimConfig(
    hand_qpos_mu=(-1.5, 0.0, 0.27, 0.27, 0.27, 0.27),
    spen_threshold=0.0,  # neighbouring fingers sit under 2 cm apart; the default margin pins the middle finger straight
)
_ABILITY_FLOATING_MOTION_PLANNER = MotionPlannerConfig(pre_grasp_hand_grad_move=1.5)
_ABILITY_TABLETOP_MOTION_PLANNER = MotionPlannerConfig(
    num_steps_pre_grasp=20,
    pre_grasp_arm_grad_move=0.03,
    pre_grasp_hand_grad_move=1.0,
)
_ABILITY_TABLETOP_EVALUATOR = EnvConfig(
    ms_env_config=MsEnvConfig(robot_uid="ability_right_floating_base"),
    load_scenes=True,
    control_freq=50,
    object_mass=1.0,
    target_lift_dist=0.08,
    num_envs=1024,
    num_perturbations=4,
    perturbation_step_range=(65, 95),
    perturbation_duration=3,
)
_ABILITY_DATASET = DatasetConfig(num_points=1024, return_object_pointcloud_only=True)
_ABILITY_TRAINER = TrainerConfig(batch_size=16, total_iters=100000, eval_freq=10000, save_freq=10000)

ability_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="ability_right_dex1b_v0_4_mini_floating",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=ability_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_ability_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=ability_right,
        motion_planner=_ABILITY_FLOATING_MOTION_PLANNER,
    ),
    grasp_optim=replace(
        _ABILITY_GRASP_OPTIM,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        # in-loop filter: heavy object + strong external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="ability_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=1.0,
            floating_object=True,
            perturbation_magnitude=250.0,
        ),
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="ability_right_floating_base"), load_scenes=True, floating_object=True
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-05-12-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=_ABILITY_DATASET,
    trainer=replace(
        _ABILITY_TRAINER,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="ability_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    enable_contact_extraction=True,
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/ability_grasps/dex1b_v0_4_floating_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/pointcloud/dex1b/ability_dex1b_v0_4_floating_mini.h5"),
    processed_data_path=str(DATA_ROOT / "sim_data/ability_grasps/dex1b_v0_4_floating_mini.h5"),
)

ability_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="ability_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=ability_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_ability_right,
        floating_base_robot_config=ability_right,
        motion_planner=_ABILITY_TABLETOP_MOTION_PLANNER,
    ),
    grasp_optim=replace(
        _ABILITY_GRASP_OPTIM,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        w_angle_z=100.0,
        evaluator=_ABILITY_TABLETOP_EVALUATOR,  # in-loop filter: external-force perturbation on
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="ability_right_floating_base"), load_scenes=True, floating_object=False
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-05-12-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=_ABILITY_DATASET,
    trainer=replace(_ABILITY_TRAINER, evaluator=replace(_ABILITY_TABLETOP_EVALUATOR, enable_perturbation=False)),
    train_single_gpu=True,
    enable_contact_extraction=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/ability_grasps/dex1b_v0_4_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/pointcloud/dex1b/ability_dex1b_v0_4_tabletop_mini.h5"),
    processed_data_path=str(DATA_ROOT / "sim_data/ability_grasps/dex1b_v0_4_tabletop_mini.h5"),
)
