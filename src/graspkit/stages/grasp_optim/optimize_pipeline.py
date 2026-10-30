"""Grasp optimization + motion planning + sim eval pipeline; CLI: scripts/stages/grasp_optim/optimize_grasps.py."""

import logging
import math
import os
import random
import shutil
from typing import Literal, Optional

import numpy as np
import torch
from rich.progress import track

from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.datasets.replay_data import ReplayData
from graspkit.envs import build_env
from graspkit.envs.base import BaseEnv
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from graspkit.stages.grasp_traj_build import build_planning_traj_builder
from graspkit.utils.io_utils import build_filesystem
from graspkit.utils.profile_utils import cuda_timeit
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.stage_utils import write_completion_marker
from graspkit.utils.tensor_utils import batchify_tensor

torch.set_float32_matmul_precision("high")
logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIM_COMPLETE_MARKER = ".optim_complete"


def resolve_robot(cfg: ExperimentConfig, robot_base: Optional[Literal["floating", "fixed"]]) -> tuple[RobotConfig, str]:
    """The optimization robot and its output dir; robot_base swaps in traj_builder's robot and suffixes the dir."""
    robot_config = cfg.robot_config
    output_dir = cfg.grasp_output_dir
    if robot_base is not None:
        tb = cfg.traj_builder
        override = tb.floating_base_robot_config if robot_base == "floating" else tb.fixed_base_robot_config
        assert override is not None, (
            f"{robot_base}-base optimization requires traj_builder.{robot_base}_base_robot_config"
        )
        if override is not robot_config:
            output_dir = output_dir.rstrip("/") + f"_{robot_base}"
        robot_config = override
    assert robot_config is not None, "optimization requires cfg.robot_config"
    assert output_dir, "optimization requires a grasp output directory"
    return robot_config, output_dir


def static_replays(
    qpos: torch.Tensor, robot_config: RobotConfig, desc: SceneDescription, stage: str, labels: list[str]
) -> list[ReplayData]:
    """(N, T, dof) optimizer qpos, object at rest; a fixed base scatters its pruned chain into the full robot."""
    qpos = qpos.cpu().numpy()
    if robot_config.base_type == "fixed":
        full_names = GraspRobot.from_config(robot_config, prune=False).actuated_joint_names
        chain = [full_names.index(n) for n in GraspRobot.from_config(robot_config).actuated_joint_names]
        full = np.tile(np.asarray(robot_config.init_qpos, dtype=np.float32), (*qpos.shape[:2], 1))
        full[..., chain] = qpos
        qpos = full
    target = desc.meshes[desc.target_actor_id]
    obj_pose = np.tile(np.array([*target.position, *target.quaternion], dtype=np.float32), (qpos.shape[1], 1))
    return [ReplayData.from_traj(q, obj_pose, robot_config, desc, stage, label) for q, label in zip(qpos, labels)]


def save_stages(
    batch: GraspKitData,
    robot_config: RobotConfig,
    optimizer: WarpGraspOptimizer,
    evaluator: Optional[BaseEnv],
    checks: dict[str, torch.Tensor],
    viz_dir: str,
) -> None:
    """Save each scene's stages for visualize_grasps.py, labeling each grasp by its first failed check or `success`."""
    diag, snaps = optimizer.sampler.diagnostics, optimizer.last_snapshots
    logger.info(f"sampler pass: { {k: f'{v.float().mean():.1%}' for k, v in diag.items() if v.dtype == torch.bool} }")
    batch = batch.to_padded()
    outcome = np.full(tuple(batch.grasp_qpos.shape[:2]), "success", dtype=object)
    for name, ok in reversed(checks.items()):  # the earliest failed check wins
        outcome[~ok.cpu().numpy()] = name
    for i, (scene_id, desc) in enumerate(zip(batch.scene_ids, batch.scene_descriptions)):
        labels, replays = outcome[i].tolist(), []
        if "ik_qpos" in diag:  # up to 16 IK candidates per verdict: `pass` or the sampler filter rejecting it
            verdict = np.full(diag["ik"].shape[1], "pass", dtype=object)
            for name in reversed(["ik"] + [f.name for f in optimizer.sampler.post_ik_filters]):
                verdict[~diag[name][i].numpy()] = f"sampler:{name}"
            keep = [c for label in dict.fromkeys(verdict) for c in np.flatnonzero(verdict == label)[:16]]
            replays += static_replays(
                diag["ik_qpos"][i, keep, None], robot_config, desc, "sample (unfiltered)", verdict[keep].tolist()
            )
        if snaps is not None:  # warp_lm records no snapshots
            replays += static_replays(snaps[i, :, :1], robot_config, desc, "sample (filtered)", labels)
            replays += static_replays(snaps[i], robot_config, desc, "optimize", labels)
        grasp = torch.stack([batch.pre_grasp_qpos[i], batch.grasp_qpos[i], batch.post_grasp_qpos[i]], dim=1)
        replays += static_replays(grasp, robot_config, desc, "grasp", labels)
        replays += static_replays(batch.full_traj_qpos[i], robot_config, desc, "motion", labels)
        if evaluator is not None:
            replays += [
                ReplayData.from_traj(r.qpos, r.obj_pose, robot_config, desc, "sim", labels[r.grasp_index])
                for r in evaluator.last_rollouts
                if r.scene_id == scene_id
            ]
        ReplayData.save(replays, f"{viz_dir}/{scene_id}.h5")


def optimize_grasps(
    cfg: ExperimentConfig,
    robot_base: Optional[Literal["floating", "fixed"]] = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    seed_everything(42)
    # Stage 1: resolve the optimization robot + output dir; skip if a previous run already finished.
    robot_config, output_dir = resolve_robot(cfg, robot_base)
    viz_dir = output_dir.rstrip("/") + ".viz"
    num_dofs = len(robot_config.init_qpos) + (6 if robot_config.base_type == "floating" else 0)
    bytes_per_grasp = 3 * num_dofs * 4

    is_local = not output_dir.startswith("s3://")
    if is_local and overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # --overwrite: wipe a prior run so it fully regenerates
    if is_local and not overwrite and os.path.exists(os.path.join(output_dir, OPTIM_COMPLETE_MARKER)):
        logger.info(f"Already complete in {output_dir}, skipping.")
        return
    if is_local:
        os.makedirs(output_dir, exist_ok=True)
    if is_local and verbose:
        os.makedirs(viz_dir, exist_ok=True)

    # Stage 2: build the traj builder, the in-loop evaluator, and the grasp optimizer.
    traj_builder = build_planning_traj_builder(cfg, device, robot_config=robot_config)
    evaluator = build_env(cfg.grasp_optim.evaluator, robot_config, device, enable_rendering=False)
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    grasp_optimizer = WarpGraspOptimizer(
        cfg.grasp_optim, robot_config, device, fixed_base_robot_config=cfg.traj_builder.fixed_base_robot_config
    )

    scene_ids = sorted(scene_descriptions.keys())
    if cfg.world_size and cfg.world_size > 1:
        scene_ids = scene_ids[cfg.rank :: cfg.world_size]  # each rank owns a disjoint slice of scenes
    # single worker: keep sorted scene order (deterministic).
    if cfg.grasp_optim.max_scenes is not None:
        scene_ids = scene_ids[: cfg.grasp_optim.max_scenes]
    threshold = cfg.grasp_optim.min_grasps_threshold
    fs = build_filesystem(output_dir)
    logger.info(f"{len(scene_ids)} scenes from {os.path.basename(cfg.scene_description_path)}")

    for attempt in range(cfg.grasp_optim.num_attempts):
        seed_everything(int.from_bytes(os.urandom(4), "big"))  # fresh samples each pass

        # Work-stealing via one detailed listing (size as a grasp-count proxy) to avoid a Ceph read storm; races ok.
        sizes = {
            os.path.splitext(os.path.basename(d["name"]))[0]: d["size"]
            for d in (fs.ls(output_dir, detail=True) if fs.exists(output_dir) else [])
        }
        todo_ids = [sid for sid in scene_ids if sizes.get(sid, 0) < threshold * bytes_per_grasp]
        # Shuffled so each batch mixes scene difficulty.
        random.shuffle(todo_ids)
        logger.info(f"Pass {attempt + 1}/{cfg.grasp_optim.num_attempts}: {len(todo_ids)}/{len(scene_ids)} scenes to do")
        if not todo_ids:
            break

        for batch_ids in track(
            batchify_tensor(todo_ids, cfg.grasp_optim.num_scenes_per_batch),
            description=f"Optimizing (pass {attempt + 1}/{cfg.grasp_optim.num_attempts})",
            total=math.ceil(len(todo_ids) / cfg.grasp_optim.num_scenes_per_batch),
        ):
            batch_descs = [scene_descriptions[sid] for sid in batch_ids]

            batch = GraspKitData(scene_ids=list(batch_ids), scene_descriptions=batch_descs)
            with cuda_timeit(f"Grasp optimization for {len(batch_descs)} scenes"):
                batch.grasp_qpos = grasp_optimizer.optimize(
                    batch.scene_descriptions, cfg.grasp_optim.num_samples_per_scene, record=verbose
                )
            batch, filter_success = traj_builder.build(batch)
            mp_success = batch.pre_traj_penetration <= 0.005

            if evaluator is not None:
                with cuda_timeit(f"Evaluation for {len(batch_descs)} scenes"):
                    eval_results = evaluator.evaluate(batch, record=verbose)
                eval_success = eval_results["success"].to(mp_success.device)
            else:
                eval_results = {}
                eval_success = torch.ones_like(mp_success)

            metric_parts = [
                f"mp={mp_success.float().mean():.2%}",
                f"eval={eval_success.float().mean():.2%}",
            ]
            for key, value in eval_results.items():
                if key != "success" and value.dtype == torch.bool:
                    metric_parts.append(f"{key}={value.float().mean():.2%}")

            final_success = filter_success & mp_success & eval_success
            if verbose:
                checks = {f"filter:{name}": mask for name, mask in traj_builder.last_masks.items()}
                checks |= {"motion": mp_success, "sim": eval_success}
                save_stages(batch, robot_config, grasp_optimizer, evaluator, checks, viz_dir)
            metric_parts.append(f"final={final_success.float().mean():.2%}")
            logger.info("Success rates: " + ", ".join(metric_parts))

            if cfg.grasp_optim.save_labels:
                labels_dir = output_dir.rstrip("/") + ".labels"
                os.makedirs(labels_dir, exist_ok=True)
                labels = {k: v.cpu() for k, v in eval_results.items() if v.dtype == torch.bool}
                torch.save(
                    {
                        "scene_ids": list(batch_ids),
                        "pre_grasp_qpos": batch.pre_grasp_qpos.cpu(),
                        "grasp_qpos": batch.grasp_qpos.cpu(),
                        "post_grasp_qpos": batch.post_grasp_qpos.cpu(),
                        "pre_traj_penetration": batch.pre_traj_penetration.cpu(),
                        "filter_success": filter_success.cpu(),
                        "mp_success": mp_success.cpu(),
                        "eval": labels,
                        "final_success": final_success.cpu(),
                    },
                    os.path.join(labels_dir, f"rank{cfg.rank}_attempt{attempt}_{batch_ids[0]}.pt"),
                )
            batch.full_traj_qpos = None
            batch.pre_traj_penetration = None
            to_save = batch.to_packed().filter(final_success.flatten())
            if to_save.num_scenes:
                to_save.save(output_dir, append=True)

    # Stage 5: shut down evaluator and write the completion marker.
    if evaluator is not None:
        evaluator.close()
    write_completion_marker(os.path.join(output_dir, OPTIM_COMPLETE_MARKER), cfg.world_size, cfg.rank)
