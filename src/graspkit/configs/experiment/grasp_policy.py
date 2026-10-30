"""Five arm/hand setups on the shared Dex1B scenes with the original ManiSkill evaluator."""

from dataclasses import replace

from graspkit.configs.env_config import CameraSpec
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
from graspkit.configs.robot import build_robot_config
from graspkit.constants import DATA_ROOT
from graspkit.utils.assembly_utils import assembled_robot_config

arx_x5_tabletop, panda_allegro_tabletop, agilex_aloha_tabletop, yam_tabletop, xarm7_sharpa_tabletop = (
    ExperimentConfig(
        name=f"{name}_tabletop",
        steps=("render_data", "optimize_grasps", "process_sim_data", "train"),
        robot_config=robot,
        traj_builder=GraspTrajBuilderConfig(
            fixed_base_robot_config=robot,
            floating_base_robot_config=build_robot_config(hand),
            plan_table_only=True,
            motion_planner=MotionPlannerConfig(pre_grasp_arm_grad_move=0.005, lifting_dist=0.15),
        ),
        grasp_optim=GraspOptimConfig(
            evaluator=EnvConfig(target_lift_dist=0.1, num_envs=128, enable_perturbation=False),
            num_samples_per_scene=64,
            num_scenes_per_batch=2,
            min_grasps_threshold=16,
            num_attempts=1,
            distance_lower=distance_lower,  # hand base to object; ARX/ALOHA's is the wrist, ~14.5 cm behind the jaws
        ),
        renderer=RendererConfig(
            num_envs=48,
            num_scenes_per_batch=16,
            num_points=2048,
            extract_segments=[],
            remove_segments=["robot"],
            points_crop_bounds=((-0.23, 0.23), (-0.23, 0.23), (0.102, 0.5)),
            camera_frame="world",
            camera_poses_path=str(DATA_ROOT / "camera_data/unified_policy/poses.json"),
            camera_intrinsic_path=str(DATA_ROOT / "camera_data/unified_policy/intrinsic.json"),
            render_downscale=2.0,
        ),
        agent=GraspPolicyConfig(
            vision_encoder="pointnet",
            cvae_latent_dim=32,
            cvae_hidden_dim=256,
            kl_weight=0.01,
            enable_kinematic_loss=True,
            pointcloud_scale=10.0,
            use_mean=False,
        ),
        dataset=DatasetConfig(num_actions=8, num_points=2048),
        trainer=TrainerConfig(evaluator=None, batch_size=32, lr=3e-4, total_iters=60000, num_dataloader_workers=4),
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini.msgpack"),
        rendered_data_path=str(DATA_ROOT / "render_data/unified_policy" / name),
        grasp_output_dir=str(DATA_ROOT / "optim_data/unified_policy" / name),
        processed_data_path=str(DATA_ROOT / "sim_data/unified_policy" / f"{name}.h5"),
    )
    for name, robot, hand, distance_lower in (
        ("arx_x5", build_robot_config("arx_x5_arm"), "arx_x5_gripper", 0.1),
        ("panda_allegro", assembled_robot_config("franka_panda_allegro_right"), "allegro_right", 0.02),
        ("agilex_aloha", build_robot_config("agilex_aloha_arm"), "agilex_aloha_gripper", 0.1),
        ("yam", assembled_robot_config("yam_yam_gripper"), "yam_gripper", 0.02),
        ("xarm7_sharpa", assembled_robot_config("xarm7_sharpa_right"), "sharpa_right", 0.02),
    )
)


# Single-arm benchmark cameras in the common tabletop frame (tabletop z=0), 320x240, fov in degrees.
# RoboDojo: env_cfg/camera/{camera_config.yml,template.py}, HF Assets/Robots/x5/robot_config.yml.
# RoboTwin: HF embodiments/aloha-agilex/config.yml and env_cfg/task_config/_camera_config.yml.
# DexJoCo: configs/multi_task/pick_bucket.yaml and xmls/{arena_arm_hand_bucket_pick,panda_allegro_copy}.xml.
CAMERAS = {
    "arx_x5": (
        CameraSpec("head", None, (0, -0.36, 0.543, 0.6123724, -0.3535534, 0.3535534, 0.6123724),
                   intrinsic=((144.065, 0, 160), (0, 168.232, 120), (0, 0, 1))),
        CameraSpec("wrist", "camera", (0, 0, 0.001, 0.9961947, 0, 0.0871558, 0),
                   intrinsic=((198.521, 0, 160), (0, 198.521, 120), (0, 0, 1))),
    ),
    "agilex_aloha": (
        CameraSpec("head", None, (-0.032, -0.45, 0.61, 0.6324555, -0.3162278, 0.3162278, 0.6324555), fov=37),
        CameraSpec("wrist", "right_camera", (0, 0, 0, 1, 0, 0, 0), fov=37),
    ),
    "panda_allegro": (
        CameraSpec("front", None, (1.45, 0, 0.68, 0, -0.2164396, 0, 0.9762961), fov=45),
        CameraSpec("wrist", "panda_link8", (-0.1202082, 0.0777817, 0.04, 0.0997343, 0.8924435, -0.3691208, 0.2394729),
                   fov=60),
    ),
    "yam": (
        CameraSpec("wrist_left", "ee_base", (0, 0.08, 0.03, 0.1925758, -0.6803783, 0.1925758, -0.6803783), fov=75),
        CameraSpec("wrist_right", "ee_base", (0, -0.08, 0.03, 0.1925758, 0.6803783, 0.1925758, 0.6803783), fov=75),
    ),
    "xarm7_sharpa": (
        CameraSpec("front", None, (0.35, -0.35, 0.55, 0.3531824, -0.3557077, 0.1473390, 0.8526578), fov=60),
        CameraSpec("wrist", "ee_right_hand_C_MC", (0.08, 0, -0.02, 0.3964244, -0.3964244, -0.5855319, -0.5855319),
                   fov=60),
    ),
}  # fmt: skip

(
    arx_x5_unified_depth,
    arx_x5_unified_rgb,
    panda_allegro_unified_depth,
    panda_allegro_unified_rgb,
    agilex_aloha_unified_depth,
    agilex_aloha_unified_rgb,
    yam_unified_depth,
    yam_unified_rgb,
    xarm7_sharpa_unified_depth,
    xarm7_sharpa_unified_rgb,
) = (
    replace(
        cfg,
        name=cfg.name.removesuffix("_tabletop") + f"_unified_{sensor}",
        robot_config=robot,
        traj_builder=replace(cfg.traj_builder, fixed_base_robot_config=robot),
        grasp_optim=replace(cfg.grasp_optim, evaluator=replace(cfg.grasp_optim.evaluator, cameras=cameras)),
        renderer=replace(
            cfg.renderer,
            env_config=replace(cfg.renderer.env_config, cameras=cameras),
            camera_poses_path=None,
            camera_intrinsic_path=None,
            render_downscale=1.0,
        ),
        agent=replace(
            cfg.agent,
            vision_encoder="pointnet" if sensor == "depth" else "rgb_resnet_views",
            generative_model="cvae",
            kl_weight=0.1,
        ),
        dataset=replace(cfg.dataset, return_object_pointcloud_only=sensor == "depth", cache_in_memory=True),
        trainer=replace(cfg.trainer, batch_size=64, total_iters=8000, save_freq=4000, num_dataloader_workers=0),
        rendered_data_path=str(data_dir / "render"),
        grasp_output_dir=str(data_dir / "grasps"),
        processed_data_path=str(data_dir / "train1024.h5"),
        ckpt_path=None,
    )
    # Official arm bases and retargeted joints preserve the world palm/wrist observation poses.
    for cfg, position, quaternion, arm_qpos in (
        (
            arx_x5_tabletop,
            (0.3, -0.4, 0.0),
            (0.707106781, 0.0, 0.0, 0.707106781),
            (0.841376126, 1.551671505, 2.204062223, -1.786368847, 0.501782715, 1.908922672),
        ),
        (
            panda_allegro_tabletop,
            (-0.6, 0.0, 0.004),
            (1.0, 0.0, 0.0, 0.0),
            (-0.021472074, 0.201649070, -0.030377362, -1.807914615, -0.023046449, 1.408620477, 0.735278726),
        ),
        (
            agilex_aloha_tabletop,
            (0.3063, -0.4185, 0.041),
            (0.703562423, 0.0, 0.0, 0.710633462),
            (0.790968955, 1.226878047, 1.545301318, -1.336650491, 0.377052426, 1.702060699),
        ),
        (
            yam_tabletop,
            yam_tabletop.robot_config.init_position,
            yam_tabletop.robot_config.init_quaternion,
            (-1.392346, 0.000417, -1.570790, 2.068998, 1.410553, 0.000357),
        ),
        (
            xarm7_sharpa_tabletop,
            xarm7_sharpa_tabletop.robot_config.init_position,
            xarm7_sharpa_tabletop.robot_config.init_quaternion,
            (1.998728, -1.393785, -1.066201, 0.756491, 5.312606, 1.185589, -1.118984),
        ),
    )
    for robot in (
        replace(
            cfg.robot_config,
            init_position=position,
            init_quaternion=quaternion,
            workspace_origin=position,
            init_qpos=arm_qpos + cfg.robot_config.init_qpos[len(arm_qpos) :],
        ),
    )
    for cameras in (CAMERAS[cfg.name.removesuffix("_tabletop")],)
    for data_dir in (
        DATA_ROOT / "benchmark/grasp_policy/unified_policy/v0.5.0_tabletop_mini" / cfg.name.removesuffix("_tabletop"),
    )
    for sensor in ("depth", "rgb")
)
