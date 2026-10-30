"""Run a grasp policy in a simulator: observe, propose, select, plan and execute, replanning mid-way in closed loop."""

import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
import torch
from robokit.xform.warp.torch_wrappers import quaternion_multiply

from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import BaseEnv
from graspkit.models.grasp_policy import GraspPolicy
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_traj_build import GraspTrajBuilder, build_planning_traj_builder
from graspkit.utils.seed_utils import seed_everything


def policy_builders(cfg: ExperimentConfig, device: torch.device) -> tuple[GraspTrajBuilder, GraspTrajBuilder, int]:
    """The policy benchmark's planner from the initial state, its replanner from the feedback step, and that step."""
    planning = replace(cfg, traj_builder=replace(cfg.traj_builder, plan_table_only=True))
    feedback_step = cfg.traj_builder.motion_planner.num_steps_pre_grasp // 2
    motion_planner = replace(
        cfg.traj_builder.motion_planner,
        num_steps_pre_grasp=cfg.traj_builder.motion_planner.num_steps_pre_grasp - feedback_step,
    )
    return (
        build_planning_traj_builder(planning, device),
        build_planning_traj_builder(planning, device, motion_planner=motion_planner),
        feedback_step,
    )


def rollout_policy(
    policy: GraspPolicy,
    cfg: ExperimentConfig,
    env: BaseEnv,
    builders: tuple[GraspTrajBuilder, GraspTrajBuilder, int],
    scenes: list[SceneDescription],
    seed: int,
    closed_loop: bool,
    selector: Literal["proposal", "learned"],
    num_samples: int,
    observer: Callable[[str, dict], None] | None = None,
) -> dict:
    """One scene per env: sim info, (b, steps, ...) commanded/measured qpos and object pose, (dx, dy, dyaw) nudges."""
    builder, feedback_builder, feedback_step = builders
    limits, root_pose = builder.robot.actuated_joint_limits_torch, builder.robot.root_pose_7d
    calibrated = cfg.renderer.camera_poses_path is not None  # calibrated views carry no initial frame
    b, ids = len(scenes), [scene.scene_id for scene in scenes]
    seeds = [int.from_bytes(hashlib.sha256(f"{seed}:{sid}".encode()).digest()[:4], "big") for sid in ids]
    rngs = [np.random.default_rng(scene_seed) for scene_seed in seeds]
    shift = np.stack([rng.uniform(-0.015, 0.015, 2) for rng in rngs])
    yaw = np.asarray([rng.uniform(-np.pi / 18, np.pi / 18) for rng in rngs])
    env.reset(scenes, init_qpos=builder.robot.init_qpos.expand(b, -1), root_pose=root_pose.expand(b, -1))
    if observer is not None:
        observer("reset", {"env": env, "scenes": scenes})
    commands, measured, objects = [], [], []
    for phase in range(2):
        if phase == 0 or closed_loop:
            current_qpos = env.get_qpos().unsqueeze(1)
            seed_everything(seeds[0] + phase)  # rendering/resampling noise of the whole chunk
            frame, poses = env.observe(), env.get_object_pose()
            views = [
                {**{key: value[i : i + 1] for key, value in frame.items()}, "qpos": current_qpos[i : i + 1]}
                for i in range(b)
            ]
            if phase == 0:
                initial = views
            chosen = []
            for i in range(b):
                if observer is not None:
                    observer(
                        "render",
                        {
                            "frame": views[i],
                            "scene": scenes[i],
                            "object_pose": poses[i],
                            "qpos": views[i]["qpos"],
                            "root_pose": root_pose,
                        },
                    )
                seed_everything(seeds[i] + phase)
                obs = policy.preprocess(initial[i], views[i], calibrated)
                with torch.no_grad():
                    actions = policy.get_action(obs, num_samples).clamp(limits[:, 0], limits[:, 1])
                    chosen.append(actions[0, 0 if selector == "proposal" else int(policy.select(obs, actions)[0])])
            chosen = torch.stack(chosen)[:, None]
            planned, _ = (builder if phase == 0 else feedback_builder).build(
                GraspKitData(
                    scene_ids=ids,
                    scene_descriptions=scenes,
                    pre_grasp_qpos=chosen[:, :, 0],
                    grasp_qpos=chosen[:, :, 1],
                    post_grasp_qpos=chosen[:, :, 2],
                ),
                start_qpos=current_qpos,
                skip_filter=True,
            )
            trajectory = planned.full_traj_qpos[:, 0]
        segment = (
            trajectory[:, :feedback_step]
            if phase == 0
            else (trajectory if closed_loop else trajectory[:, feedback_step:])
        )
        for step in range(segment.shape[1]):
            env.step_qpos(segment[:, step])
            commands.append(segment[:, step])
            measured.append(env.get_qpos())
            objects.append(env.get_object_pose())
            if observer is not None:
                observer("measured", {"env": env})
        if phase == 0:
            pose = env.get_object_pose().clone()
            pose[:, :2] += pose.new_tensor(shift)
            rotation = np.stack((np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)), axis=1)
            pose[:, 3:] = quaternion_multiply(pose.new_tensor(rotation), pose[:, 3:])
            env.set_object_pose(pose)
    info = env.get_info()
    if observer is not None:
        observer("result", {"env": env, "scenes": scenes, "info": info})
    return {
        "info": info,
        "qpos": torch.stack(commands, 1).cpu().numpy(),
        "measured_qpos": torch.stack(measured, 1).cpu().numpy(),
        "object_pose": torch.stack(objects, 1).cpu().numpy(),
        "displacement": np.concatenate((shift, yaw[:, None]), axis=1),
    }
