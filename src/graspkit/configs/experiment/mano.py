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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_6drot, pointnet_basic
from graspkit.configs.robot.g1_mano import g1_mano_right
from graspkit.configs.robot.mano import mano_right
from graspkit.configs.robot.xarm7_mano import xarm7_mano_right
from graspkit.constants import DATA_ROOT

_MANO_GRASP_OPTIM = GraspOptimConfig(
    hand_qpos_mu=(0.0,) * 16,
)
_MANO_FLOATING_MOTION_PLANNER = MotionPlannerConfig(
    num_steps_pre_grasp=0,
    num_steps_post_grasp=0,
    pre_grasp_hand_grad_move=1.0,
    post_grasp_hand_grad_move=2.0,
)
_MANO_TABLETOP_EVALUATOR = EnvConfig(
    ms_env_config=MsEnvConfig(robot_uid="mano_right_floating_base"),
    load_scenes=True,
    control_freq=50,
    object_mass=0.01,
    target_lift_dist=0.08,
    perturbation_magnitude=0.5,
)
# g1_mano experiment: same evaluator on the fixed-base g1 agent.
_G1_MANO_TABLETOP_EVALUATOR = replace(
    _MANO_TABLETOP_EVALUATOR,
    ms_env_config=replace(_MANO_TABLETOP_EVALUATOR.ms_env_config, robot_uid="g1_mano_right"),
)
_MANO_DATASET = DatasetConfig(num_points=1024, return_object_pointcloud_only=True)
_MANO_TRAINER = TrainerConfig(batch_size=16, total_iters=10000, eval_freq=10000, save_freq=10000)

mano_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="mano_right_dex1b_v0_4_mini_floating",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=mano_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_mano_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=mano_right,
        contact_avoid_filter=True,
        motion_planner=_MANO_FLOATING_MOTION_PLANNER,
    ),
    grasp_optim=replace(
        _MANO_GRASP_OPTIM,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        sampler_type="floating_base",
        # in-loop filter: external-force perturbation to stress-test grasps
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="mano_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            perturbation_magnitude=0.5,
        ),
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="mano_right_floating_base"), load_scenes=True, floating_object=True
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-05-12-camera_pose.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=_MANO_DATASET,
    trainer=replace(
        _MANO_TRAINER,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="mano_right_floating_base"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/mano_grasps/dex1b_v0_4_floating_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/pointcloud/dex1b/mano_dex1b_v0_4_floating_mini.h5"),
    processed_data_path=str(DATA_ROOT / "sim_data/mano_traj/dex1b_v0_4_floating_mini.h5"),
)

mano_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="mano_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=mano_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_mano_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=mano_right,
        contact_avoid_filter=True,
        motion_planner=_MANO_FLOATING_MOTION_PLANNER,
    ),
    grasp_optim=replace(
        _MANO_GRASP_OPTIM,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        w_angle_z=100.0,
        sampler_type="floating_base",
        evaluator=_MANO_TABLETOP_EVALUATOR,  # in-loop filter: external-force perturbation on
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="mano_right_floating_base"), load_scenes=True, floating_object=False
        ),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=_MANO_DATASET,
    trainer=replace(_MANO_TRAINER, evaluator=replace(_MANO_TABLETOP_EVALUATOR, enable_perturbation=False)),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/mano_grasps/dex1b_v0_4_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/pointcloud/dex1b/mano_dex1b_v0_4_tabletop_mini.h5"),
    processed_data_path=str(DATA_ROOT / "sim_data/mano_traj/dex1b_v0_4_tabletop_mini.h5"),
)

g1_mano_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="g1_mano_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=g1_mano_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=g1_mano_right,
        floating_base_robot_config=g1_mano_right,  # policy predicts fixed-base qpos; planned as-is
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(),
    ),
    grasp_optim=replace(
        _MANO_GRASP_OPTIM,
        optimizer_backend="warp_gd",
        evaluator=_G1_MANO_TABLETOP_EVALUATOR,  # in-loop filter: external-force perturbation on
    ),
    agent=pointnet_basic,
    dataset=_MANO_DATASET,
    trainer=replace(_MANO_TRAINER, evaluator=replace(_G1_MANO_TABLETOP_EVALUATOR, enable_perturbation=False)),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/g1_mano_grasps/dex1b_v0_4_tabletop_mini"),
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="g1_mano_right")),
        camera_poses_path=str(calib.calib_file("g1/26-02-12-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("g1/25-10-07-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/g1_26-02-12"),
    processed_data_path=str(DATA_ROOT / "sim_data/g1_mano_traj/dex1b_v0_4_tabletop_mini.h5"),
)
