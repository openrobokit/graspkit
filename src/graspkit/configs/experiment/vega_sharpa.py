from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_5_0
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
from graspkit.configs.grasp_policy import pointnet_0_1_kl_kin_loss
from graspkit.configs.robot.vega_sharpa import vega_sharpa_right
from graspkit.constants import DATA_ROOT

# Fixed-base right-arm + Sharpa: grasp optim -> head-cam render -> open-loop policy, over dex1b v0.5 tabletop_mini.
vega_sharpa_right_dex1b_v0_5_mini_zed = ExperimentConfig(
    name="vega_sharpa_right_dex1b_v0_5_mini_zed",
    steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
    robot_config=vega_sharpa_right,
    traj_builder=GraspTrajBuilderConfig(
        fixed_base_robot_config=vega_sharpa_right,
        floating_base_robot_config=vega_sharpa_right,  # optimized fixed-base grasps planned as-is
        pen_filter=True,
        pen_filter_threshold=0.002,
        # Low grasps zigzag under the default L-BFGS budget (~15 reversals on deploy); 5x/20x/20x converges to sub-mm.
        motion_planner=MotionPlannerConfig(
            backend="warp",
            lifting_dist=0.15,
            traj_opt_max_iter=500,
            traj_opt_smoothness_weight=4000.0,
            traj_opt_accel_smoothness_weight=600.0,
            close_during_holding=True,
        ),
    ),
    grasp_optim=GraspOptimConfig(
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="vega_sharpa_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.1,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=512,
            num_seeds=2,
            enable_perturbation=True,
            perturbation_magnitude=5.0,
            perturbation_step_range=(-25, -5),
            num_perturbations=4,
        ),  # in-loop grasp filter
        sampler_type="wbc",  # humanoid-over-table; alt "fixed_base"
        # Natural right wrist: fingers point forward-left, so the hand reaches in from the front-right.
        hand_axis="four_fingers_up",
        hand_axis_target=(1.0, 1.0, 0.0),
        filter_hand_axis_max_angle_deg=45.0,
        init_rest_pose_max_dist=2.0,
        # optimizer frees right arm + right hand (the topology default)
        distance_lower=0.08,
        distance_upper=0.30,
        w_spen=20.0,
        spen_threshold=0.0,
        spen_representation="sphere",
        num_samples_per_scene=256,
        num_scenes_per_batch=4,  # 1024 grasps/batch; 16 OOMs the warp motion planner on 24GB
        min_grasps_threshold=1,
        # No max_scenes: the first sorted scene ids are all test-split, so any small cap yields zero training data.
    ),
    agent=pointnet_0_1_kl_kin_loss,
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True),
    trainer=TrainerConfig(
        batch_size=256,
        evaluator=EnvConfig(
            ms_env_config=MsEnvConfig(robot_uid="vega_sharpa_right"),
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            contact_offset=0.005,
            num_envs=512,
            enable_perturbation=False,
        ),
    ),
    train_single_gpu=True,
    # dexmate variant of tabletop_mini: +-15cm table height, +-25cm object region, 0.6x0.6 table.
    scene_description_path=str(dex1b_v0_5_0.DEXMATE_TABLETOP_MINI_PATH),
    grasp_output_dir=str(DATA_ROOT / "optim_data/vega_sharpa_grasps/dex1b_v0_5_mini_right"),
    # Head-mounted ZED; 26-08-21 calibration FK'd to the base frame at rest pose -- recalibrate if the head moves.
    renderer=RendererConfig(
        env_config=EnvConfig(ms_env_config=MsEnvConfig(robot_uid="vega_sharpa_right"), camera_near=0.05),
        camera_poses_path=str(calib.calib_file("vega_sharpa/26-08-21-finger-camera_extrinsic.json")),
        camera_intrinsic_path=str(calib.calib_file("vega_sharpa/26-08-05-camera_intrinsic.json")),
    ),
    rendered_data_path=str(DATA_ROOT / "render_data/dex1b_v0.5.0/vega_sharpa_right_26-08-05"),
    processed_data_path=str(DATA_ROOT / "sim_data/vega_sharpa_traj/dex1b_v0_5_mini_right.h5"),
)
