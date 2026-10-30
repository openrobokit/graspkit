"""Label grasp-policy states in any simulator into resumable HDF5 shards, then assemble shards into a training bank."""

import graspkit  # noqa: F401 I001

import collections
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import tyro
from robokit.xform.warp.torch_wrappers import quaternion_multiply
from torch.nn import functional as F

from graspkit.configs.env_config import IsaacEnvConfig, MjNativeEnvConfig
from graspkit.configs.experiment import build_experiment_config
from graspkit.constants import DATA_ROOT
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env
from graspkit.models.checkpoint import load_policy
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.policy_rollout import policy_builders
from graspkit.utils.io_utils import build_filesystem
from graspkit.utils.seed_utils import seed_everything


def main(
    stage: Literal["label", "assemble"],
    robot: Literal["arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"],
    scenes: str = str(DATA_ROOT / "runs/retrain_20261015/common/train.msgpack"),
    gt_dir: str = "",
    output: str = "",
    simulator: Literal["maniskill", "isaac", "mujoco"] = "maniskill",
    checkpoint: Path | None = None,
    gt: int = 16,
    proposals: int = 64,
    scenes_per_batch: int = 32,
    num_envs: int = 2560,
    rank: int = 0,
    world_size: int = 1,
    isaac_command: tuple[str, ...] = ("/isaac-sim/python.sh",),
    shards: tuple[str, ...] = (),
    bank: Path | None = None,
    validation_objects: int = 32,
):
    """`label` reads GT from `--gt-dir` and writes shards to `--output`; `assemble` merges `--shards` into `--bank`."""
    descriptions = SceneDescription.load_scene_descriptions_msgpack(scenes, return_splits=True)["train"]
    if stage == "assemble":
        bank.mkdir(parents=True, exist_ok=True)
        seen, positives, trials, hidden = (collections.Counter() for _ in range(4))
        aliases, replay_ids, replay_actions, replay_rewards = {}, [], [], []
        total, squared, count = 0.0, 0.0, 0
        with h5py.File(bank / "data.h5", "w") as out:
            for root in shards:
                fs = build_filesystem(root)
                for path in sorted(fs.glob(f"{root.rstrip('/')}/*.h5")):
                    with h5py.File(io.BytesIO(fs.cat_file(path))) as shard:
                        assert shard.attrs["robot"] == robot
                        sim = shard.attrs["simulator"]
                        for name, group in shard.items():
                            sid, phase = name.rsplit("__", 1)
                            candidates, success = group["candidates"][:], group["success"][:]
                            policy = group["source"][:] == b"policy"
                            for source, mask in (("gt", ~policy), ("policy", policy)):
                                positives[f"{sim}/{phase}/{source}"] += int(success[mask].sum())
                                trials[f"{sim}/{phase}/{source}"] += int(mask.sum())
                            repeat = seen[sid, sim, phase]  # a later round of the same scene and simulator
                            seen[sid, sim, phase] += 1
                            state = f"{sid}__{sim}{repeat or ''}__{phase}"
                            obs = group["obs"]
                            if not (obs["pointcloud_segmentation"][:] == obs["target_actor_segmentation_id"][()]).any():
                                hidden[f"{sim}/{phase}"] += 1  # the robot hides the object; depth input is empty
                                continue
                            aliases[state] = descriptions[sid]
                            entry = out.create_group(f"data/{state}")
                            shard.copy(obs, entry)
                            if success.any():
                                entry.create_dataset("actions", data=candidates[success], compression="gzip")
                                total += candidates[success].sum(0, dtype=np.float64)
                                squared += np.square(candidates[success], dtype=np.float64).sum(0)
                                count += int(success.sum())
                            if policy.sum() == 64:
                                replay_ids.append(state)
                                replay_actions.append(candidates[policy].reshape(64, -1))
                                replay_rewards.append(success[policy])
            mean = total / count
            out.create_dataset("meta/action_mean", data=mean.astype(np.float32))
            out.create_dataset(
                "meta/action_std", data=np.sqrt(np.maximum(squared / count - mean**2, 0)).astype(np.float32)
            )
        meshes = {state: aliases[state].meshes[aliases[state].target_actor_id].collision_mesh_path for state in aliases}
        held = sorted(
            {meshes[state] for state in replay_ids},
            key=lambda mesh: hashlib.sha256(f"scorer:{mesh}".encode()).hexdigest(),
        )[:validation_objects]
        validation = [state for state in replay_ids if meshes[state] in held]
        if replay_ids:  # GT-only rounds have no policy proposals to score
            torch.save(
                {
                    "scene_ids": replay_ids,
                    # train.py re-encodes these with the actor's 512 features
                    "features": torch.zeros(len(replay_ids), 1, 512),
                    "actions": torch.from_numpy(np.stack(replay_actions)),
                    "rewards": torch.from_numpy(np.stack(replay_rewards)).float(),
                    "scorer_validation_ids": validation,
                    "reserved_validation_ids": [],
                },
                bank / "replay.pt",
            )
        SceneDescription.save_scene_descriptions_msgpack({"train": aliases, "test": {}}, str(bank / "scenes.msgpack"))
        rates = {key: [positives[key] / trials[key], trials[key]] for key in sorted(trials) if trials[key]}
        summary = {"bank": str(bank), "states": len(aliases), "positives": count, "replay": len(replay_ids)}
        print(json.dumps(summary | {"validation": len(validation), "skipped_hidden_target": hidden}))
        print(json.dumps({"positive_rate_and_trials": rates}), flush=True)
        return
    device = torch.device("cuda:0")
    cfg = build_experiment_config(f"{robot}_unified_depth")
    policy = None
    if checkpoint is not None:
        policy, _, saved = load_policy(checkpoint, str(device))
        assert saved["experiment"] in (f"{robot}_unified_depth", f"{robot}_unified_rgb")
        policy.cfg = replace(policy.cfg, grasp_evaluator_guidance_steps=0)
    evaluator = cfg.grasp_optim.evaluator
    if simulator == "isaac":
        evaluator = replace(evaluator, isaac_env_config=IsaacEnvConfig(worker_command=list(isaac_command)))
        assert max(num_envs, scenes_per_batch) <= evaluator.isaac_env_config.max_envs
    if simulator == "mujoco":
        evaluator = replace(evaluator, mj_native_env_config=MjNativeEnvConfig())
    observer = build_env(
        replace(evaluator, num_envs=scenes_per_batch), cfg.robot_config, device, renderer_cfg=cfg.renderer
    )
    runner = build_env(
        replace(evaluator, num_envs=num_envs),
        cfg.robot_config,
        device,
        enable_rendering=False,
        renderer_cfg=cfg.renderer,
    )
    builder, feedback_builder, feedback_step = policy_builders(cfg, device)
    root, limits = builder.robot.root_pose_7d, builder.robot.actuated_joint_limits_torch
    gt_fs, out_fs = build_filesystem(gt_dir), build_filesystem(output)
    available = {Path(name).stem for name in gt_fs.ls(gt_dir, detail=False) if name.endswith(".h5")}
    ids = sorted(descriptions)[rank::world_size]  # batches stay fixed for resuming; run once the GT is complete
    print(json.dumps({"scenes": len(ids), "without_gt": len(set(ids) - available)}), flush=True)
    for index, begin in enumerate(range(0, len(ids), scenes_per_batch)):
        path = f"{output.rstrip('/')}/{rank:04d}-{world_size:04d}-{index:05d}.h5"
        chunk = [sid for sid in ids[begin : begin + scenes_per_batch] if sid in available]
        if not chunk or out_fs.exists(path):
            continue
        started, simulated = time.monotonic(), 0.0
        b, batch = len(chunk), [descriptions[sid] for sid in chunk]
        fill = [*range(b)] + [0] * (scenes_per_batch - b)  # observer envs beyond the batch repeat its first scene
        seeds = [int.from_bytes(hashlib.sha256(f"collect:{sid}".encode()).digest()[:4], "big") for sid in chunk]
        rngs = [np.random.default_rng(scene_seed) for scene_seed in seeds]
        shift = np.stack([rng.uniform(-0.015, 0.015, 2) for rng in rngs])  # the benchmark's displacement
        yaw = np.asarray([rng.uniform(-np.pi / 18, np.pi / 18) for rng in rngs])
        rotation = np.stack((np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)), axis=1)
        gt_grasps = []
        for sid, rng in zip(chunk, rngs):
            loaded = GraspKitData.load(f"{gt_dir.rstrip('/')}/{sid}.h5")
            grasps = torch.stack((loaded.pre_grasp_qpos, loaded.grasp_qpos, loaded.post_grasp_qpos), 1).to(device)
            gt_grasps.append(grasps[rng.choice(len(grasps), min(gt, len(grasps)), replace=False)])
        first, states, rates = [None] * b, {}, collections.defaultdict(list)
        start_qpos, start_pose = builder.robot.init_qpos.expand(b, -1), None  # feedback: the driver's state
        for phase, phase_name in enumerate(("initial", "feedback")):
            observer.reset(
                [batch[i] for i in fill],
                init_qpos=start_qpos[fill],
                root_pose=root.expand(len(fill), -1),
            )
            if phase == 1:
                observer.set_object_pose(start_pose[fill])
            qpos = observer.get_qpos()[:b].clamp(limits[:, 0], limits[:, 1])  # Isaac rejects resets past the limits
            seed_everything(seeds[0] + phase)
            frame = {key: value[:b].cpu() for key, value in observer.observe().items()}
            candidates = []
            for i in range(b):
                proposed = gt_grasps[i][:0]
                if policy is not None:
                    seed_everything(seeds[i] + phase)
                    view = {**{key: value[i : i + 1] for key, value in frame.items()}, "qpos": qpos[i : i + 1, None]}
                    if phase == 0:
                        first[i] = view
                    obs = policy.preprocess(first[i], view, False)
                    with torch.no_grad():
                        proposed = policy.get_action(obs, proposals).clamp(limits[:, 0], limits[:, 1])[0]
                candidates.append(torch.cat((gt_grasps[i], proposed)))
            counts = [len(grasps) for grasps in candidates]
            grid = torch.stack([torch.cat((c, c[-1:].expand(max(counts) - len(c), -1, -1))) for c in candidates])
            planned, _ = (builder if phase == 0 else feedback_builder).build(
                GraspKitData(
                    scene_ids=chunk,
                    scene_descriptions=batch,
                    pre_grasp_qpos=grid[:, :, 0],
                    grasp_qpos=grid[:, :, 1],
                    post_grasp_qpos=grid[:, :, 2],
                ),
                start_qpos=qpos[:, None].expand(-1, max(counts), -1),
                skip_filter=True,
            )
            owner = [i for i, n in enumerate(counts) for _ in range(n)]
            trajectories = torch.cat([planned.full_traj_qpos[i, :n] for i, n in enumerate(counts)])
            padding = -len(owner) % num_envs  # runner envs beyond the rollouts repeat its first one
            owner, trajectories = (
                owner + owner[:1] * padding,
                torch.cat((trajectories, trajectories[:1].repeat(padding, 1, 1))),
            )
            success, measured, moved = [], [], []
            simulated -= time.monotonic()
            for offset in range(0, len(owner), num_envs):
                rows = owner[offset : offset + num_envs]
                runner.reset([batch[i] for i in rows], init_qpos=qpos[rows], root_pose=root.expand(num_envs, -1))
                if phase == 1:
                    runner.set_object_pose(start_pose[rows])
                for step in range(trajectories.shape[1]):
                    if phase == 0 and step == feedback_step:
                        measured.append(runner.get_qpos().clamp(limits[:, 0], limits[:, 1]))
                        pose = runner.get_object_pose().clone()
                        pose[:, :2] += pose.new_tensor(shift[rows])
                        pose[:, 3:] = quaternion_multiply(pose.new_tensor(rotation[rows]), pose[:, 3:])
                        runner.set_object_pose(pose)
                        moved.append(pose)
                    runner.step_qpos(trajectories[offset : offset + num_envs, step])
                success.append(runner.get_info()["success"].cpu())
            simulated += time.monotonic()
            success = torch.cat(success)[: sum(counts)].numpy()
            starts = np.cumsum([0, *counts])
            images = F.interpolate(  # the policy's own RGB resize, stored as uint8
                frame["rgb"].flatten(0, 1).permute(0, 3, 1, 2).float(),
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            )
            images = images.round().clamp(0, 255).byte().permute(0, 2, 3, 1).unflatten(0, (b, -1))
            if phase == 0:
                driver = [  # the closed loop continues from policy proposal 0, else from a seeded GT grasp
                    starts[i] + (len(gt_grasps[i]) if policy is not None else rngs[i].integers(len(gt_grasps[i])))
                    for i in range(b)
                ]
                start_qpos, start_pose = torch.cat(measured)[driver], torch.cat(moved)[driver]
                qpos0, cameras0, images0, points0 = qpos, frame["camera_poses"], images, (frame["xyz"], frame["seg"])
            for i, sid in enumerate(chunk):
                source = np.array(["gt"] * len(gt_grasps[i]) + ["policy"] * (counts[i] - len(gt_grasps[i])), "S6")
                states[f"{sid}__{phase_name}"] = {
                    "obs/pointcloud": torch.cat((points0[0][i], frame["xyz"][i]))[None].float().numpy(),
                    "obs/pointcloud_segmentation": torch.cat((points0[1][i], frame["seg"][i]))[None]
                    .to(torch.int16)
                    .numpy(),
                    "obs/target_actor_segmentation_id": np.int64(int(frame["seg_id"][i])),
                    "obs/rgb": torch.cat((images0[i], images[i])).numpy(),
                    "obs/camera_poses": torch.cat((cameras0[i], frame["camera_poses"][i]))[:, None].float().numpy(),
                    "obs/qpos": torch.stack((qpos0[i], qpos[i])).float().cpu().numpy(),
                    "candidates": candidates[i].float().cpu().numpy(),
                    "success": success[starts[i] : starts[i + 1]],
                    "source": source,
                    "displacement": np.array([*shift[i], yaw[i]], np.float32),
                }
                for name in ("gt", "policy"):
                    rates[f"{phase_name}_{name}"].extend(
                        success[starts[i] : starts[i + 1]][source == name.encode()].tolist()
                    )
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as shard:
            shard.attrs.update(robot=robot, simulator=simulator, checkpoint=str(checkpoint or ""))
            for state, entry in states.items():
                for key, value in entry.items():
                    shard.create_dataset(f"{state}/{key}", data=value, compression="gzip" if np.ndim(value) else None)
        out_fs.pipe_file(path + ".part", buffer.getvalue())
        out_fs.mv(path + ".part", path)
        rollouts = sum(len(entry["success"]) for entry in states.values())
        row = {key: [float(np.mean(values)), len(values)] for key, values in rates.items() if values}
        print(
            json.dumps(
                {
                    "shard": path,
                    "scenes": b,
                    "rollouts": rollouts,
                    "seconds": time.monotonic() - started,
                    "simulated_seconds": simulated,
                    "rollouts_per_simulated_second": rollouts / simulated,
                    "success_rate_and_trials": row,
                }
            ),
            flush=True,
        )
    observer.close()
    runner.close()


if __name__ == "__main__":
    tyro.cli(main)
