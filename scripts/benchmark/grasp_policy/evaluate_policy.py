"""Checkpoint and tabletop policy benchmarks using GraspKit's existing pipeline."""

import graspkit  # noqa: I001

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import torch
import trimesh
import tyro
from robokit.xform.numpy import quaternion_raw_multiply
from robokit.xform.utils import coord_conversion
from robokit.xform.warp.torch_wrappers import matrix_to_quaternion, pose7_to_tf_mat
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

from graspkit.configs.env_config import MsEnvConfig
from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import EnvConfig, ExperimentConfig, RendererConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot import build_robot_config
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env, build_renderer
from graspkit.models.checkpoint import load_policy
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy
from graspkit.scene.scene_description import MeshDescription, SceneDescription
from graspkit.stages.grasp_filters import compute_penetration_depth
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from graspkit.stages.grasp_traj_build import GraspTrajBuilder, build_planning_traj_builder
from graspkit.stages.policy_rollout import policy_builders, rollout_policy
from graspkit.utils.seed_utils import seed_everything

_ROBOTS = {
    "arx_x5": ("arx_x5_tabletop", "arx_x5_gripper"),
    "panda_allegro": ("panda_allegro_tabletop", "allegro_right"),
    "agilex_aloha": ("agilex_aloha_tabletop", "agilex_aloha_gripper"),
    "yam": ("yam_tabletop", "yam_gripper"),
    "xarm7_sharpa": ("xarm7_sharpa_tabletop", "sharpa_right"),
}
_EYES = ((0.35, 0.0, 0.4), (-0.25, 0.3, 0.4), (-0.25, -0.3, 0.4))


def main(
    output: Path,
    checkpoint: Path | None = None,
    seed: int = 0,
    split: Literal["train", "test"] = "test",
    ids_path: Path | None = None,
    observations_path: Path | None = None,
    require_native: bool = False,
    require_object_disjoint: bool = False,
    evaluation_experiment: str | None = None,
    mode: Literal[
        "checkpoint", "pc_open_loop", "pc_closed_loop", "rgb_closed_loop", "policy_open_loop", "policy_closed_loop"
    ] = "checkpoint",
    robot: Literal["arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"] | None = None,
    arm_mode: Literal["floating_ik", "arm_aware"] = "floating_ik",
    rank: int = 0,
    world_size: int = 1,
    start_index: int = 0,
    max_scenes: int = 0,
    num_samples: int = 32,
    num_candidates: int = 16,
    num_iters: int = 10000,
    selector: Literal["proposal", "perception_sim", "learned"] = "proposal",
    pregrasp_distance: float = 0.005,
    scene_description_path: Path | None = None,
    guidance_steps: int = 0,
    num_envs: int = 1,
    trajectories: Path | None = None,
    policy: tyro.conf.Suppress[GraspPolicy | None] = None,
    config: tyro.conf.Suppress[ExperimentConfig | None] = None,
    observer: tyro.conf.Suppress[Callable[[str, dict], None] | None] = None,
) -> None:
    """Execute one selected grasp per scene; set CUDA_VISIBLE_DEVICES, plus PYTHONHASHSEED=0 in checkpoint mode."""
    assert 0 <= rank < world_size and num_samples >= num_candidates > 0
    graspkit.link()
    assert output.suffix == (".json" if mode == "checkpoint" else ".jsonl") and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if mode in ("policy_open_loop", "policy_closed_loop"):
        assert checkpoint is not None and selector != "perception_sim"
        seed_everything(seed)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = (
            config
            if config is not None
            else build_experiment_config(
                saved["experiment"] if evaluation_experiment is None else evaluation_experiment
            )
        )
        if policy is None:
            policy, _loaded_config, saved = load_policy(checkpoint, str(device))
        assert cfg.robot_config.base_type == "fixed" and not policy.cfg.convert_rxyz_to_rot6d
        policy.cfg = replace(policy.cfg, grasp_evaluator_guidance_steps=guidance_steps)
        scenes = SceneDescription.load_scene_descriptions_msgpack(
            str(scene_description_path) if scene_description_path is not None else cfg.scene_description_path,
            return_splits=True,
        )[split]
        ids = sorted(scenes) if ids_path is None else sorted(json.loads(ids_path.read_text()))
        assert set(ids) <= set(scenes)
        ids = ids[rank::world_size][start_index:]
        if max_scenes:
            ids = ids[:max_scenes]
        builders = policy_builders(cfg, device)
        builder, _, feedback_step = builders
        env = build_env(
            replace(cfg.grasp_optim.evaluator, num_envs=num_envs), cfg.robot_config, device, renderer_cfg=cfg.renderer
        )
        dumped = []
        with output.open("w") as result_file:
            for start in range(0, len(ids), num_envs):  # one env per scene; each scene keeps its own seeds
                started = time.monotonic()
                chunk = ids[start : start + num_envs]
                result = rollout_policy(
                    policy,
                    cfg,
                    env,
                    builders,
                    [scenes[sid] for sid in chunk],
                    seed,
                    mode == "policy_closed_loop",
                    selector,
                    num_samples,
                    observer,
                )
                for i, sid in enumerate(chunk):
                    row = {
                        "scene_id": sid,
                        "robot": cfg.name.split("_unified_")[0].removesuffix("_tabletop"),
                        "input": "rgb" if policy.cfg.vision_encoder.startswith("rgb") else "depth",
                        "mode": mode,
                        "model": policy.cfg.generative_model,
                        "selector": selector,
                        "guidance_steps": policy.cfg.grasp_evaluator_guidance_steps,
                        "seed": seed,
                        "split": split,
                        "training_scenes": len(saved["training"]["train_ids"]),
                        "proposals": num_samples,
                        "attempts": 1,
                        "feedback_step": feedback_step,
                        "observation_count": 1 + (mode == "policy_closed_loop"),
                        "control_steps": cfg.traj_builder.motion_planner.num_steps,
                        "success": bool(result["info"]["success"][i]),
                        "is_lifted": float(result["info"]["is_lifted"][i]),
                        "is_strict_grasped": float(result["info"]["is_strict_grasped"][i]),
                        "displacement": result["displacement"][i].tolist(),
                        "seconds": (time.monotonic() - started) / len(chunk),
                    }
                    result_file.write(json.dumps(row) + "\n")
                    result_file.flush()
                    print(json.dumps(row), flush=True)
                    dumped.append(
                        {
                            "scene_id": sid,
                            "qpos": result["qpos"][i],
                            "measured_qpos": result["measured_qpos"][i],
                            "object_pose": result["object_pose"][i],
                            "feedback_step": feedback_step,
                            "displacement": row["displacement"],
                            "root_pose": builder.robot.root_pose_7d.cpu().numpy(),
                            "init_qpos": builder.robot.init_qpos.cpu().numpy(),
                            "joint_names": list(builder.robot.actuated_joint_names),
                            "control_freq": cfg.grasp_optim.evaluator.control_freq,
                            "success": row["success"],
                        }
                    )
        if trajectories is not None:
            trajectories.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dumped, trajectories)
        env.close()
        return
    with torch.set_grad_enabled(mode != "checkpoint"):
        if mode == "checkpoint":
            assert checkpoint is not None and robot is None
            assert os.environ["PYTHONHASHSEED"] == "0" and not output.with_suffix(".pt").exists()
            seed_everything(seed)
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            cfg = build_experiment_config(
                saved["experiment"] if evaluation_experiment is None else evaluation_experiment
            )
            policy_cfg = GraspPolicyConfig(**saved["policy_config"])
            if require_native:
                assert saved["training"]["initialization"] == "random"
                assert saved["training"]["initial_checkpoint"] is None
                assert saved["training"]["right_weights"] is False and saved["training"]["teacher_transfer"] is False
                assert policy_cfg.grasp_evaluator_guidance_steps == 0
                assert not policy_cfg.convert_rxyz_to_rot6d or policy_cfg.analytic_euler
                assert cfg.trainer.evaluator.ms_env_config.robot_uid == "sharpa_left_floating_base"
                assert cfg.trainer.evaluator.num_seeds == 1
            assert cfg.robot_config.base_type == "floating"
            policy = build_grasp_policy(policy_cfg, cfg.robot_config, device).eval()
            policy.load_state_dict(saved["agent"], strict=True)
            observations = str(observations_path) if observations_path is not None else cfg.processed_data_path
            dataset = GraspDataset(
                replace(cfg.dataset, return_actions=False), observations, cfg.scene_description_path, split
            )
            splits = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path, return_splits=True)
            if ids_path is not None:
                assert split == "train"
                selected = set(json.loads(ids_path.read_text()))
                assert selected <= set(dataset.scene_ids)
                dataset.scene_ids = sorted(selected)
            else:
                assert set(dataset.scene_ids) == set(splits[split])
                dataset.scene_ids = sorted(dataset.scene_ids)
            dataset.scene_ids = dataset.scene_ids[rank::world_size]
            assert 0 <= start_index <= len(dataset.scene_ids)
            dataset.scene_ids = dataset.scene_ids[start_index:]
            if max_scenes:
                dataset.scene_ids = dataset.scene_ids[:max_scenes]
            assert not set(dataset.scene_ids) & (
                set(saved["training"]["train_ids"]) | set(saved["training"]["validation_ids"])
            )
            all_descriptions = {**splits["train"], **splits["test"]}
            training_meshes = {
                all_descriptions[sid].meshes[all_descriptions[sid].target_actor_id].collision_mesh_path
                for sid in saved["training"]["train_ids"]
            }
            evaluation_meshes = {
                all_descriptions[sid].meshes[all_descriptions[sid].target_actor_id].collision_mesh_path
                for sid in dataset.scene_ids
            }
            if require_object_disjoint or (require_native and split == "test"):
                assert not training_meshes & evaluation_meshes
            loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
            ids, predictions = [], []
            modules = [("encoder", policy.vision_encoder), ("decoder", policy.generator.decoder)]
            if policy_cfg.cvae_learned_prior:
                modules.append(("prior", policy.generator.prior_encoder))
            forward_calls = {name: [] for name, _ in modules}
            hooks = [
                module.register_forward_hook(
                    lambda module, inputs, output, name=name: forward_calls[name].append(output.shape[0])
                )
                for name, module in modules
            ]
            batch_sizes = []
            seed_everything(seed)
            torch.cuda.synchronize()
            start_time = time.monotonic()
            for batch in loader:
                obs = {key: value.to(device) for key, value in batch["observations"].items()}
                with torch.inference_mode(require_native), torch.no_grad():
                    predictions.append(policy.get_action(obs, num_samples_per_scene=1))
                batch_sizes.append(len(batch["metas"]["scene_id"]))
                ids.extend(batch["metas"]["scene_id"])
            for hook in hooks:
                hook.remove()
            if require_native:
                assert all(calls == batch_sizes for calls in forward_calls.values())
            torch.cuda.synchronize()
            prediction_seconds = time.monotonic() - start_time
            actions = torch.cat(predictions)
            assert ids == dataset.scene_ids and torch.isfinite(actions).all()
            builder = GraspTrajBuilder(cfg.traj_builder, device)
            trajectories, depths = [], []
            for start in range(0, len(ids), 16):
                batch_ids = ids[start : start + 16]
                q = actions[start : start + 16]
                descriptions = [dataset.scene_descriptions[sid] for sid in batch_ids]
                batch, _ = builder.build(
                    GraspKitData(
                        scene_ids=batch_ids,
                        scene_descriptions=descriptions,
                        pre_grasp_qpos=q[..., 0, :],
                        grasp_qpos=q[..., 1, :],
                        post_grasp_qpos=q[..., 2, :],
                    )
                )
                trajectories.append(batch.full_traj_qpos)
                depths.append(compute_penetration_depth(descriptions, q[..., 1, :], builder.robot))
            trajectories, penetration = torch.cat(trajectories), torch.cat(depths).flatten()
            torch.save(
                {"scene_ids": ids, "actions": actions.cpu(), "trajectories": trajectories.cpu()},
                output.with_suffix(".pt"),
            )
            env_cfg = replace(cfg.trainer.evaluator, num_envs=16)
        else:
            assert robot is not None and checkpoint is None and evaluation_experiment is None
            assert ids_path is None and observations_path is None
            assert not require_native and not require_object_disjoint
            experiment_name, hand_name = _ROBOTS[robot]
            cfg = build_experiment_config(experiment_name)
            cfg = replace(cfg, grasp_optim=replace(cfg.grasp_optim, num_iters=num_iters))
            scenes = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path, return_splits=True)[
                split
            ]
            ids = sorted(scenes)[rank::world_size]
            assert 0 <= start_index <= len(ids)
            ids = ids[start_index:]
            if max_scenes:
                ids = ids[:max_scenes]

            eyes = torch.tensor(_EYES, dtype=torch.float32, device=device)
            target = torch.tensor((0.0, 0.0, 0.05), dtype=torch.float32, device=device).expand_as(eyes)
            forward = torch.nn.functional.normalize(target - eyes, dim=-1)
            right = torch.nn.functional.normalize(
                torch.cross(forward, torch.tensor((0.0, 0.0, 1.0), device=device).expand_as(eyes), dim=-1), dim=-1
            )
            up = torch.cross(forward, right, dim=-1)
            sapien2opencv = coord_conversion("sapien", "opencv", return_tensors="pt").to(device)
            rotation = torch.stack((right, up, forward), dim=-1) @ sapien2opencv
            camera_poses = torch.cat((eyes, matrix_to_quaternion(rotation)), dim=-1).unsqueeze(0)
            root = torch.tensor((100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0), device=device)

            observation_config = EnvConfig(
                ms_env_config=MsEnvConfig(robot_uid="franka_gripper"),
                camera_width=640,
                camera_height=480,
                camera_intrinsic=((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)),
            )
            renderer = build_renderer(
                RendererConfig(
                    env_config=observation_config,
                    num_envs=3,
                    num_points=2048,
                    obs_mode="rgb" if mode == "rgb_closed_loop" else "pointcloud",
                ),
                build_robot_config("franka_gripper"),
                device,
            )
            robot_qpos = renderer.robot_helper.init_qpos.to(device).view(1, 1, -1)
            projections = []
            grid = torch.empty((0, 3), device=device)
            if mode == "rgb_closed_loop":
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(-0.3, 0.3, 0.005, device=device),
                        torch.arange(-0.3, 0.3, 0.005, device=device),
                        torch.arange(-0.1, 0.3, 0.005, device=device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 3)
                cv_frame = torch.eye(4, device=device)
                cv_frame[:3, :3] = sapien2opencv
                for pose in pose7_to_tf_mat(camera_poses[0]):
                    world_to_camera = cv_frame @ torch.linalg.inv(pose)
                    point_camera = world_to_camera[:3, :3] @ grid.T + world_to_camera[:3, 3:4]
                    u = (500.0 * point_camera[0] / point_camera[2] + 320.0).round().long()
                    v = (500.0 * point_camera[1] / point_camera[2] + 240.0).round().long()
                    valid = (point_camera[2] > 0) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
                    projections.append((valid, u.clamp(0, 639), v.clamp(0, 479)))

            floating_config = build_robot_config(hand_name)
            floating_optimizer = WarpGraspOptimizer(cfg.grasp_optim, floating_config, device)
            proposal_builder = build_planning_traj_builder(
                cfg,
                device,
                motion_planner=replace(
                    cfg.traj_builder.motion_planner, backend="torch", pre_grasp_arm_grad_move=pregrasp_distance
                ),
                robot_config=floating_config,
            )
            floating_builder = GraspTrajBuilder(
                replace(
                    cfg.traj_builder,
                    floating_base_robot_config=floating_config,
                    convert_to_fixed=True,
                    plan_table_only=True,
                    arm_range_fraction=1.0,
                    ori_threshold_deg=180.0,
                ),
                device,
            )
            arm_optimizer = (
                WarpGraspOptimizer(cfg.grasp_optim, cfg.robot_config, device, fixed_base_robot_config=cfg.robot_config)
                if arm_mode == "arm_aware"
                else None
            )
            arm_builder = (
                build_planning_traj_builder(
                    replace(cfg, traj_builder=replace(cfg.traj_builder, plan_table_only=True)), device
                )
                if arm_mode == "arm_aware"
                else None
            )

            env_cfg = replace(cfg.grasp_optim.evaluator, num_envs=num_samples)

        env = build_env(env_cfg, cfg.robot_config, device)
        batch_size = len(ids) if mode == "checkpoint" else 1
        with TemporaryDirectory(prefix="graspkit_policy_") as mesh_directory, output.open("w") as result_file:
            for start in range(0, len(ids), batch_size):
                started = time.monotonic()
                batch_ids = ids[start : start + batch_size]
                attempt = None
                if mode == "checkpoint":
                    attempt = GraspKitData(
                        scene_ids=batch_ids,
                        scene_descriptions=[dataset.scene_descriptions[sid] for sid in batch_ids],
                        grasp_qpos=actions[:, :, 1, :],
                        full_traj_qpos=trajectories,
                    )
                else:
                    sid = batch_ids[0]
                    scene = scenes[sid]
                    digest = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
                    scene_seed = int.from_bytes(digest[:4], "big")
                    np.random.seed(scene_seed)
                    torch.manual_seed(scene_seed)
                    rng = np.random.default_rng(scene_seed)
                    displacement = rng.uniform(-0.015, 0.015, 2)
                    yaw = np.deg2rad(rng.uniform(-10.0, 10.0))
                    mesh = scene.meshes[scene.target_actor_id]
                    moved_mesh = replace(
                        mesh,
                        position=(
                            mesh.position[0] + displacement[0],
                            mesh.position[1] + displacement[1],
                            mesh.position[2],
                        ),
                        quaternion=tuple(
                            quaternion_raw_multiply(
                                np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), np.asarray(mesh.quaternion)
                            )
                        ),
                        bounds=None,
                    )
                    execution_scene = replace(scene, meshes={**scene.meshes, scene.target_actor_id: moved_mesh})
                    observed_scene = scene if mode == "pc_open_loop" else execution_scene
                    row = {
                        "scene_id": sid,
                        "robot": robot,
                        "mode": mode,
                        "arm_mode": arm_mode,
                        "arm_init": cfg.grasp_optim.sampler_type if arm_mode == "arm_aware" else None,
                        "selector": selector,
                        "split": split,
                        "success": False,
                        "attempts": 0,
                        "stage": "observation",
                        "displacement": [*displacement.tolist(), float(yaw)],
                    }

                    if mode == "rgb_closed_loop":
                        observed_mesh = observed_scene.meshes[observed_scene.target_actor_id]
                        object_pose = torch.tensor(
                            (*observed_mesh.position, *observed_mesh.quaternion), device=device, dtype=torch.float32
                        )
                        images = renderer.render(
                            [observed_scene],
                            camera_poses=camera_poses,
                            robot_root_poses=root,
                            robot_qpos=robot_qpos.expand(1, 2, -1),
                            object_poses=torch.stack((object_pose, root)).view(1, 2, 7),
                        )["sensor_data"]["base_camera"]["rgb"].numpy()
                        keep = torch.ones(len(grid), dtype=torch.bool, device=device)
                        for camera_images, (valid, u, v) in zip(images, projections):
                            difference = np.abs(
                                camera_images[0].astype(np.int16) - camera_images[1].astype(np.int16)
                            ).max(axis=-1)
                            mask = torch.as_tensor(binary_dilation(difference > 15, iterations=2), device=device)
                            keep &= valid & mask[v, u]
                        points = grid[keep].cpu().numpy()
                    else:
                        cloud = (
                            renderer.render(
                                [observed_scene],
                                camera_poses=camera_poses,
                                robot_root_poses=root,
                                robot_qpos=robot_qpos,
                            )["pointcloud"]["xyz"][:, 0]
                            .reshape(-1, 3)
                            .numpy()
                        )
                        points = cloud[np.linalg.norm(cloud, axis=1) > 0]
                    points = np.unique(np.round(points, 4), axis=0)
                    if len(points) >= 4 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 3:
                        if len(points) > 4096:
                            points = points[rng.choice(len(points), 4096, replace=False)]
                        table = scene.boxes["table"]
                        table_plane = (0.0, 0.0, 1.0, -(table.position[2] + table.size[2] / 2))
                        table_points = points.copy()
                        table_points[:, 2] = -table_plane[3]
                        observed_mesh = trimesh.PointCloud(np.concatenate((points, table_points))).convex_hull
                        mesh_path = Path(mesh_directory) / f"{scene_seed}.glb"
                        observed_mesh.export(mesh_path)
                        perception_scene = SceneDescription(
                            scene_id=sid,
                            target_actor_id="observed",
                            boxes={"table": table},
                            meshes={"observed": MeshDescription("observed", str(mesh_path), (1.0, 1.0, 1.0))},
                        )
                        if arm_mode == "arm_aware":
                            refined = arm_optimizer.optimize([perception_scene], num_samples)
                            converted, _ = arm_builder.build(
                                GraspKitData(scene_ids=[sid], scene_descriptions=[perception_scene], grasp_qpos=refined)
                            )
                            valid = converted.pre_traj_penetration <= 0.005
                            candidate_indices = torch.arange(num_samples, device=device)
                            row["num_proposals"] = num_samples
                        else:
                            proposals = floating_optimizer.optimize([perception_scene], num_samples)[:, :num_candidates]
                            proposal = proposal_builder.motion_planner.plan(
                                GraspKitData(
                                    scene_ids=[sid],
                                    scene_descriptions=[perception_scene],
                                    grasp_qpos=proposals,
                                ),
                                return_pre_post_only=True,
                            )
                            converted, ik_ok = floating_builder.build(proposal)
                            row["num_proposals"] = proposals.shape[1]
                            row["num_ik"] = int(ik_ok.sum())
                            row["min_ik_penetration"] = (
                                float(converted.pre_traj_penetration[ik_ok].min()) if ik_ok.any() else None
                            )
                            valid = ik_ok & (converted.pre_traj_penetration <= 0.005)
                            candidate_indices = torch.arange(proposals.shape[1], device=device)
                        row["num_planned"] = int(valid.sum())
                        row["stage"] = "plan"
                        if valid.any():
                            selected = int(valid[0].nonzero()[0])
                            if selector == "perception_sim":
                                options = valid[0].nonzero().flatten()
                                prediction = env.evaluate(
                                    GraspKitData(
                                        scene_ids=[sid],
                                        scene_descriptions=[perception_scene],
                                        pre_grasp_qpos=converted.pre_grasp_qpos[:, options],
                                        grasp_qpos=converted.grasp_qpos[:, options],
                                        post_grasp_qpos=converted.post_grasp_qpos[:, options],
                                        full_traj_qpos=converted.full_traj_qpos[:, options],
                                    )
                                )
                                hits = prediction["success"][0].nonzero().flatten()
                                selected = int(options[int(prediction["success"][0].float().argmax())])
                                row["model_evaluations"] = len(options)
                                row["predicted_successes"] = len(hits)
                            row["candidate_index"] = int(candidate_indices[selected])
                            attempt = GraspKitData(
                                scene_ids=[sid],
                                scene_descriptions=[execution_scene],
                                pre_grasp_qpos=converted.pre_grasp_qpos[:, selected : selected + 1],
                                grasp_qpos=converted.grasp_qpos[:, selected : selected + 1],
                                post_grasp_qpos=converted.post_grasp_qpos[:, selected : selected + 1],
                                full_traj_qpos=converted.full_traj_qpos[:, selected : selected + 1],
                            )
                if attempt is not None:
                    result = env.evaluate(attempt)
                    if mode == "checkpoint":
                        raw = result["success"].flatten()
                        joint = raw & (penetration <= 0.005)
                        summary = {"trial": 0, "raw": int(raw.sum()), "joint": int(joint.sum())}
                        rows = {
                            sid: {
                                "success": bool(raw[i]),
                                "penetration": float(penetration[i]),
                                "joint_success": bool(joint[i]),
                            }
                            for i, sid in enumerate(ids)
                        }
                    else:
                        row["attempts"] = 1
                        row["stage"] = "lift"
                        row["success"] = bool(result["success"][0, 0])
                        row["is_lifted"] = bool(result["is_lifted"][0, 0])
                        row["is_strict_grasped"] = bool(result["is_strict_grasped"][0, 0])
                if mode != "checkpoint":
                    row["seconds"] = time.monotonic() - started
                    result_file.write(json.dumps(row) + "\n")
                    result_file.flush()
                    print(json.dumps({"scene_id": sid, "success": row["success"], "stage": row["stage"]}), flush=True)

            if mode == "checkpoint":
                report = {
                    "checkpoint": str(checkpoint),
                    "experiment": saved["experiment"],
                    "evaluation_experiment": cfg.name,
                    "policy_config": asdict(policy_cfg),
                    "environment": asdict(env_cfg),
                    "seed": seed,
                    "hash_seed": "0",
                    "split": split,
                    "scenes": len(ids),
                    "num_envs": 16,
                    "batch_size": 16,
                    "num_workers": 4,
                    "num_points": cfg.dataset.num_points,
                    "observations_path": observations,
                    "all_predictions_before_simulation": True,
                    "samples_per_scene": 1,
                    "native_contract_verified": require_native,
                    "training_meshes": len(training_meshes),
                    "evaluation_meshes": len(evaluation_meshes),
                    "shared_meshes_with_training": len(training_meshes & evaluation_meshes),
                    "object_disjoint_required": require_object_disjoint or (require_native and split == "test"),
                    "forward_call_batch_sizes": forward_calls,
                    "prediction_seconds": prediction_seconds,
                    "trials": [{"summary": summary, "rows": rows}],
                }
                result_file.write(json.dumps(report) + "\n")
                print(json.dumps({"output": str(output), "scenes": len(ids), **summary}), flush=True)
        env.close()
        if mode != "checkpoint":
            renderer.close()


if __name__ == "__main__":
    tyro.cli(main)
