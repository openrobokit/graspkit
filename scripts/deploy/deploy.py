"""Interactive capture -> segment -> plan -> deploy; the experiment owns policy/robots/camera, --env picks sim or real.

    uv run --extra sam2 scripts/deploy/deploy.py \
        --env sim \
        --exp-config-name vega_sharpa_right_dex1b_v0_5_mini_zed \
        --ckpt-path logs/train/vega_sharpa_right_dex1b_v0_5_mini/checkpoints/best_eval_success_rate.pt

    uv run --extra sam2 --extra xarm scripts/deploy/deploy.py \
        --env real \
        --exp-config-name xhand_right_dex1b_v0_5_mini_tabletop_fixed_eval \
        --ckpt-path <checkpoint> \
        --real-env-config-name xarm7_xhand_right

    uv run --extra sam2 scripts/deploy/deploy.py \
        --env sim \
        --exp-config-name xhand_right_dex1b_v0_5_mini_tabletop \
        --ckpt-path logs/train/xhand_right_dex1b_v0_5_mini_tabletop/checkpoints/best_eval_success_rate.pt \
        --convert-to-fixed True

"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import json
import time
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch
import tyro
from visualization_helper import DeployVisualizer

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import MsEnv
from graspkit.models.grasp_policy import build_grasp_policy
from graspkit.scene.scene_description import SceneDescription, get_table_only_scene_descriptions
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_traj_build import GraspTrajBuilder
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.xform_utils import compute_camera_world_pose_7d


@dataclass
class Args:
    exp_config_name: str
    """Experiment supplying the policy, robots, camera, trajectory builder, and sim scenes."""
    ckpt_path: str | None = None
    """Policy checkpoint; None uses the trainer's output convention for the experiment."""
    env: Literal["sim", "real"] = "sim"
    """Observation and qpos execution backend."""
    real_env_config_name: str | None = None
    """Hardware adapter from REAL_ENV_CONFIGS (--env real); None uses the experiment's value."""
    scene_id: str | None = None
    """Initial sim scene. The Viser dropdown can switch among all loaded scenes."""
    num_scenes: int | None = None
    """Limit the sim scene dropdown to the first N scenes."""
    num_samples: int = 256
    sam_model: str = "facebook/sam2.1-hiera-base-plus"
    """Hugging Face SAM2 image model used by the Viser prompt interface."""
    convert_to_fixed: bool | None = None
    """Enable floating-hand to fixed-robot IK when the experiment does not already configure it."""
    workspace_center: tuple[float, float, float] | None = None
    """Real-camera crop center in the robot-base frame; None uses the RealEnvConfig value."""
    pc_offset: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    """Offset applied to the centered policy point cloud."""
    post_grasp_hand_close_scale: float = -0.5
    """Extra multiple of grasp->post-grasp hand closure; -0.5 keeps half of it."""
    deploy_batch_size: int = -1
    """Qpos steps per Deploy click; -1 executes the whole trajectory."""
    loop: bool = False
    """Return to Capture after deployment, allowing sim scene switching."""
    camera_poses_path: str | None = None
    """Camera extrinsic JSON with ``T_base_camera``; None uses the experiment's renderer value."""
    auto: bool = False
    """Advance Viser stages automatically for validation."""


if __name__ == "__main__":
    seed_everything(42)
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = build_experiment_config(args.exp_config_name)
    floating = exp.robot_config.base_type == "floating"

    ckpt_path = args.ckpt_path or f"logs/train/{exp.name}/checkpoints/best_eval_success_rate.pt"
    agent = build_grasp_policy(exp.agent, exp.robot_config, device)
    agent.load_state_dict(torch.load(ensure_local_filepath(ckpt_path), map_location=device)["agent"])
    agent.eval()

    traj_builder_cfg = exp.traj_builder
    if floating and args.convert_to_fixed and not traj_builder_cfg.convert_to_fixed:
        traj_builder_cfg = replace(
            traj_builder_cfg,
            convert_to_fixed=True,
            motion_planner=MotionPlannerConfig(post_grasp_hand_grad_move=0.1, num_steps_pre_to_grasp=40),
        )
    traj_builder = GraspTrajBuilder(traj_builder_cfg, device)
    robot_helper = traj_builder.robot
    init_qpos = robot_helper.init_qpos.to(device)
    frozen_indices = torch.ones(init_qpos.shape[0], dtype=torch.bool, device=device)
    frozen_indices[robot_helper.arm_joint_indices] = frozen_indices[robot_helper.hand_joint_indices] = False

    camera_poses_path = args.camera_poses_path or exp.renderer.camera_poses_path
    with open(camera_poses_path) as f:
        t_base_camera = torch.tensor(json.load(f)["T_base_camera"], device=device, dtype=torch.float32)
    print(f"camera extrinsic: {camera_poses_path}")

    if args.env == "sim":
        assert exp.trainer.evaluator is not None
        evaluator_cfg = replace(exp.trainer.evaluator, num_envs=1)
        if floating and args.convert_to_fixed and not exp.traj_builder.convert_to_fixed:
            evaluator_cfg = replace(
                evaluator_cfg,
                ms_env_config=replace(
                    evaluator_cfg.ms_env_config, robot_uid=traj_builder_cfg.fixed_base_robot_config.name
                ),
            )
        with open(exp.renderer.camera_intrinsic_path) as f:
            intr = json.load(f)["intrinsic"]
        downscale = exp.renderer.render_downscale
        evaluator_cfg = replace(
            evaluator_cfg,
            camera_width=int(intr[5] / downscale),
            camera_height=int(intr[4] / downscale),
            camera_intrinsic=(
                (intr[0] / downscale, 0, intr[2] / downscale),
                (0, intr[1] / downscale, intr[3] / downscale),
                (0, 0, 1),
            ),
            camera_near=exp.renderer.env_config.camera_near,
        )
        split = "train" if exp.trainer.eval_on_train else "test"
        scenes = SceneDescription.load_scene_descriptions_msgpack(exp.scene_description_path, return_splits=True)[split]
        scene_ids = list(scenes)
        if args.scene_id is not None:
            assert args.scene_id in scenes
            scene_ids.remove(args.scene_id)
            scene_ids.insert(0, args.scene_id)
        scene_ids = scene_ids[: args.num_scenes]
        assert scene_ids
        scenes = {scene_id: scenes[scene_id] for scene_id in scene_ids}

        if exp.renderer.camera_frame == "robot":
            camera_robot = traj_builder_cfg.fixed_base_robot_config
            anchor = torch.tensor(
                (*camera_robot.init_position, *camera_robot.init_quaternion), device=device, dtype=torch.float32
            )
        else:
            anchor = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), device=device)
        camera_pose = compute_camera_world_pose_7d(anchor, t_base_camera.unsqueeze(0))
        capture_root_pose = torch.tensor((100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32)
        deploy_root_pose = robot_helper.root_pose_7d
        env = MsEnv(
            evaluator_cfg,
            robot_config=robot_helper.config,
            device=device,
            obs_mode="pointcloud",
            renderer_cfg=exp.renderer,
        )
    else:
        assert traj_builder.robot.config.base_type == "fixed", "real deploy requires fixed-base trajectories"
        assert exp.renderer.camera_frame == "robot", "real deploy calibration must be robot-base relative"
        from graspkit.configs.real_env import REAL_ENV_CONFIGS
        from graspkit.envs import RealEnv

        real_env_config_name = args.real_env_config_name or exp.real_env_config_name
        assert real_env_config_name is not None, "pass --real-env-config-name or set it on the experiment"
        real_env_config = REAL_ENV_CONFIGS[real_env_config_name]
        env = RealEnv(
            real_env_config,
            device,
            t_robot_cam=t_base_camera.cpu().numpy(),
            workspace_center=args.workspace_center,
        )
        assert env.robot_helper.actuated_joint_names == robot_helper.actuated_joint_names
        time.sleep(2.0)
        size = real_env_config.table_size
        scene = get_table_only_scene_descriptions(
            table_size=size,
            table_pos=(*real_env_config.table_xy, real_env_config.table_z - size[2] / 2),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
        scenes = {scene.scene_id: scene}
        camera_pose = None
        capture_root_pose = deploy_root_pose = robot_helper.root_pose_7d

    assert not args.auto or args.env == "sim", "--auto uses simulator ground truth to choose a SAM2 prompt"
    viz = DeployVisualizer(
        robot_helper,
        traj_builder.filter_robot,
        device,
        scenes=scenes,
        auto=args.auto,
        sam_model=args.sam_model,
        observe_cloud=env.get_observed_cloud if args.env == "real" else None,
        read_qpos=env.get_qpos_cpu if args.env == "real" else None,
    )
    offset = torch.tensor(args.pc_offset, dtype=torch.float32, device=device)

    with torch.no_grad():
        while True:
            viz.wait("Capture")
            scene = scenes[viz.scene_id]
            env.reset(
                scene=scene,
                init_qpos=init_qpos,
                root_pose=capture_root_pose,
                camera_pose=camera_pose,
            )
            rgb = env.get_rgb()
            viz.show_capture(rgb)

            if not viz.wait("Segment"):
                continue
            gt_mask = env.get_gt_mask() if args.env == "sim" else None
            auto_prompt = None
            if args.auto:
                pixels = torch.from_numpy(gt_mask).nonzero()
                center = pixels.float().mean(dim=0)
                row, col = pixels[torch.argmin(torch.square(pixels - center).sum(dim=1))]
                auto_prompt = (int(col), int(row))
            mask = viz.segment(rgb, auto_prompt)
            if gt_mask is not None:
                intersection = (mask & gt_mask).sum()
                union = (mask | gt_mask).sum()
                print(f"SAM2/GT mask IoU: {intersection / union:.3f}")
            pts = env.get_pointcloud(mask)
            assert pts.shape[1] > 0, "empty segmented point cloud"
            print(f"segmented cloud: {pts.shape[1]} points, z [{pts[0, :, 2].min():.3f}, {pts[0, :, 2].max():.3f}]")
            viz.show_segment(scene, pts)

            if not viz.wait("Check Grasp/Trajectory"):
                continue
            center = pts.mean(dim=1, keepdim=True)
            if floating:
                actions = agent.get_action(
                    {"pointcloud": pts - center + offset}, num_samples_per_scene=args.num_samples
                )[0]
                qpos = torch.cat([actions[..., :3] + center, actions[..., 3:]], dim=-1)
            else:
                qpos = agent.get_action({"pointcloud": pts}, num_samples_per_scene=args.num_samples)[0]
                # Pin uncommanded joints: policy drift (vega torso ~0.04 rad) makes the plan's FK mismatch the robot.
                qpos[..., frozen_indices] = init_qpos[frozen_indices]
            hand_indices = slice(6, None) if floating else robot_helper.hand_joint_indices
            qpos[:, 2, hand_indices] += (
                qpos[:, 2, hand_indices] - qpos[:, 1, hand_indices]
            ) * args.post_grasp_hand_close_scale
            batch = GraspKitData(
                scene_ids=[scene.scene_id],
                scene_descriptions=[scene],
                pre_grasp_qpos=qpos[:, 0][None],
                grasp_qpos=qpos[:, 1][None],
                post_grasp_qpos=qpos[:, 2][None],
            )
            planned, valid = traj_builder.build(
                batch,
                start_qpos=init_qpos,
                skip_filter=True,
                target_xy=center[0, :, :2],
            )
            valid_indices = torch.where(valid[0])[0][:8]
            if valid_indices.shape[0] == 0:
                assert args.loop, "no valid grasp trajectory"
                print("no valid grasp trajectory; returning to Capture (switch scene or re-segment)")
                continue
            viz.show_plan(qpos[:, 1], planned.grasp_qpos[0, valid_indices], planned.full_traj_qpos[0, valid_indices])
            print(
                f"{scene.scene_id}: {pts.shape[1]} points, {int(valid.sum())}/{valid.numel()} valid, "
                f"showing {valid_indices.shape[0]} candidates; select one via the 'Grasp' dropdown"
            )

            env.reset(
                scene=scene,
                init_qpos=init_qpos,
                root_pose=deploy_root_pose,
                camera_pose=camera_pose,
            )
            frame_delay = 1 / 30 if args.env == "sim" else 0.0
            # Deploy in slices: "Deploy Until Step" pauses mid-trajectory to check the observed cloud against the URDF.
            abandoned = False
            done = 0
            full_traj = viz.traj
            while done < full_traj.shape[0]:
                if not viz.wait("Deploy"):
                    abandoned = True
                    break
                selected = int(valid_indices[int(viz.grasp_slider.value)])
                until = min(int(viz.deploy_until.value) + 1, full_traj.shape[0])
                if until <= done:
                    print(f"'Deploy Until Step' is {until - 1}, already at {done - 1}; raise it to advance")
                    continue
                print(f"deploying grasp #{selected}, steps {done}..{until - 1} of {full_traj.shape[0]}")
                if args.env == "real":
                    # One firmware-interpolated trajectory per phase; the hand ramps while the arm is near-stationary.
                    mp = traj_builder_cfg.motion_planner
                    edges = np.cumsum(
                        [
                            0,
                            mp.num_steps_pre_grasp,
                            mp.num_steps_pre_to_grasp,
                            mp.num_steps_holding,
                            mp.num_steps_post_grasp,
                        ]
                    )
                    for a, b in zip(edges[:-1], edges[1:]):
                        a, b = max(a, done), min(b, until)
                        if a >= b:
                            continue
                        infos = env.set_qpos_traj(
                            full_traj[a:b],
                            lambda step, qpos, cmd=None, offset=int(a): viz.show_deploy_qpos(
                                offset + step, qpos, 0.0, commanded=cmd
                            ),
                        )
                else:
                    batch_size = until - done if args.deploy_batch_size == -1 else args.deploy_batch_size
                    for start in range(done, until, batch_size):
                        if start > done:
                            viz.wait("Deploy", abortable=False)
                        infos = env.set_qpos_traj(
                            full_traj[start : min(start + batch_size, until)],
                            lambda step, qpos, offset=start, delay=frame_delay: viz.show_deploy_qpos(
                                offset + step, qpos, delay, env.get_object_pose()[0]
                            ),
                        )
                done = until
            if abandoned:
                continue
            print(f"deploy complete: {infos}")

            # Open the hand in place (drops the object), then park on a collision-checked plan instead of a blind reset.
            viz.wait("Release", abortable=False)
            release = full_traj[-1].clone()
            release[robot_helper.hand_joint_indices] = init_qpos[robot_helper.hand_joint_indices]
            env.step_qpos(release)
            planner = traj_builder.motion_planner
            # Park against a table raised 5 cm: the optimizer rides the margin, leaving ~2 cm that tracking lag can eat.
            park_scene = scene
            if args.env == "real":
                table = scene.boxes["table"]
                park_scene = get_table_only_scene_descriptions(
                    table_size=tuple(table.size),
                    table_pos=(table.position[0], table.position[1], table.position[2] + 0.05),
                    table_quat=tuple(table.quaternion),
                )
            park = planner.interpolate_qpos(release[None, None], init_qpos[None, None], num_steps=2)
            park, _ = planner.optimize_trajs(park, SceneGeom([park_scene], device=device))
            # No reset: the park ends at rest, and a reset would crawl out settling at 0.2 rad/s; Capture resets anyway.
            env.set_qpos_traj(planner.interpolate_trajs(park, 150)[0, 0])
            if not args.loop:
                break

    env.close()
    viz.close()
