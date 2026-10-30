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
from graspkit.configs.robot.xarm7_xhand import xarm7_xhand_right
from graspkit.configs.robot.xhand import xhand_right
from graspkit.constants import DATA_ROOT

xhand_right_dex1b_v0_4_mini_floating = ExperimentConfig(
    name="xhand_right_dex1b_v0_4_mini_floating",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=xhand_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_xhand_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=xhand_right,
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            num_steps_post_grasp=0,
            pre_grasp_hand_grad_move=0.2,
            post_grasp_hand_grad_move=0.1,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.0,  # the thumb and index sit under 1 cm apart; any margin holds the index off the object
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        sampler_type="floating_base",
        # in-loop filter stress-tests with a heavy object + external force; robot_uid is a static base (no shaky PD)
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=1.0,
            floating_object=True,
        ),
    ),
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right_floating_base"), load_scenes=True, floating_object=True
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
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            floating_object=True,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_4.FLOATING_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xhand_grasps/dex1b_v0_4_floating_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4/xhand"),
    processed_data_path=str(DATA_ROOT / "sim_data/xhand_floating/dex1b_v0_4_floating_mini.h5"),
)

# Tabletop base: floating-base optimizer with the object on the table; `tabletop_dev` etc. `replace` this.
xhand_right_dex1b_v0_4_mini_tabletop = ExperimentConfig(
    name="xhand_right_dex1b_v0_4_mini_tabletop",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=xhand_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=xarm7_xhand_right,  # arm the camera is mounted on (camera_frame="robot")
        floating_base_robot_config=xhand_right,
        contact_avoid_filter=True,
        motion_planner=MotionPlannerConfig(
            num_steps_pre_grasp=0,
            num_steps_post_grasp=100,
            pre_grasp_hand_grad_move=0.2,
            post_grasp_hand_grad_move=0.1,
            lifting_dist=0.20,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.0,  # the thumb and index sit under 1 cm apart; any margin holds the index off the object
        w_angle_z=100.0,
        # in-loop filter: external-force perturbation on
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right_floating_base"),
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
    renderer=RendererConfig(
        env_config=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right_floating_base"), load_scenes=True, floating_object=False
        ),
        camera_poses_path=str(calib.calib_file("xarm/26-04-27-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("xarm/25-02-19-camera_intrinsic.json")),
    ),
    agent=pointnet_0_1_kl_6drot,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=16,
        total_iters=40000,
        eval_freq=10000,
        save_freq=10000,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="xhand_right_floating_base"),
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
    grasp_output_dir=str(DATA_ROOT / "optim_data/xhand_grasps/dex1b_v0_4_tabletop_mini"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.4_tabletop_mini/xhand"),
    processed_data_path=str(DATA_ROOT / "sim_data/xhand_floating/dex1b_v0_4_tabletop_mini.h5"),
)

# v0.5 tabletop: same floating-base recipe as v0.4 tabletop_mini, on the v0.5.0 `tabletop` scene set.
xhand_right_dex1b_v0_5_mini_tabletop = replace(
    xhand_right_dex1b_v0_4_mini_tabletop,
    name="xhand_right_dex1b_v0_5_mini_tabletop",
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xhand_grasps/dex1b_v0_5_tabletop_mini"),
    renderer=replace(
        xhand_right_dex1b_v0_4_mini_tabletop.renderer,
        camera_poses_path=str(calib.calib_file("xarm/26-05-13-camera_pose.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0_tabletop_mini/xhand"),
    processed_data_path=str(DATA_ROOT / "sim_data/xhand_floating/dex1b_v0_5_tabletop_mini.h5"),
)

# Grasp-traj-filter test: optim/train stay floating (first-64-scenes subset); trainer eval converts to fixed-base.
xhand_right_dex1b_v0_5_mini_tabletop_fixed_eval = replace(
    xhand_right_dex1b_v0_5_mini_tabletop,
    name="xhand_right_dex1b_v0_5_mini_tabletop_fixed_eval",
    traj_builder=replace(
        xhand_right_dex1b_v0_5_mini_tabletop.traj_builder,
        convert_to_fixed=True,
        motion_planner=MotionPlannerConfig(post_grasp_hand_grad_move=0.1, num_steps_pre_to_grasp=40),
    ),
    # eval_on_train: the first-64-scenes subset has no test-split scenes; eval robot is the fixed xarm7_xhand_right.
    trainer=replace(
        xhand_right_dex1b_v0_5_mini_tabletop.trainer,
        evaluator=replace(
            xhand_right_dex1b_v0_5_mini_tabletop.trainer.evaluator,
            ms_env_config=replace(
                xhand_right_dex1b_v0_5_mini_tabletop.trainer.evaluator.ms_env_config, robot_uid="xarm7_xhand_right"
            ),
        ),
        total_iters=20000,
        eval_freq=5000,
        save_freq=5000,
        eval_on_train=True,
    ),
    scene_description_path=str(DATA_ROOT / "scene_description/dex1b_v0_5_tabletop_mini_first64.msgpack"),
    grasp_output_dir=str(DATA_ROOT / "optim_data/xhand_grasps/dex1b_v0_5_tabletop_mini_fixed_eval"),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0_5_tabletop_mini_fixed_eval/xhand"),
    processed_data_path=str(DATA_ROOT / "sim_data/xhand_floating/dex1b_v0_5_tabletop_mini_fixed_eval.h5"),
)
