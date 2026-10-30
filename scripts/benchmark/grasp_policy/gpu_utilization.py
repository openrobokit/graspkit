"""Compare simulation batches using identical native grasp trajectories, seed, and evaluation settings."""

import json
import resource
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import torch
import tyro


def main(
    stage: Literal["prepare", "run"],
    payload: Path,
    output: Path,
    experiment: str = "g1_inspire_right_wbc_dex1b_v0_5_small_s3",
    scene_path: Path | None = None,
    scene_ids_path: Path | None = None,
    num_scenes: int = 1,
    num_samples: int = 4096,
    num_envs: int = 1024,
    repeats: int = 2,
    enable_rendering: bool = True,
) -> None:
    from graspkit.configs.experiment import build_experiment_config
    from graspkit.datasets.graspkit_data import GraspKitData
    from graspkit.envs import build_env
    from graspkit.scene.scene_description import SceneDescription
    from graspkit.stages.grasp_optim import WarpGraspOptimizer
    from graspkit.stages.grasp_traj_build import build_planning_traj_builder
    from graspkit.utils.io_utils import ensure_local_filepath
    from graspkit.utils.seed_utils import seed_everything

    assert num_scenes > 0 and num_samples > 0 and num_envs > 0 and repeats > 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_everything(0)
    cfg = deepcopy(build_experiment_config(experiment))
    cfg.grasp_optim.evaluator.ms_env_config.reconfiguration_freq = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    if stage == "prepare":
        scene_path = Path(ensure_local_filepath(str(scene_path or cfg.scene_description_path)))
        scenes = SceneDescription.load_scene_descriptions_msgpack(str(scene_path), return_splits=True)["train"]
        scene_ids = json.loads(scene_ids_path.read_text()) if scene_ids_path is not None else sorted(scenes)
        assert len(scene_ids) >= num_scenes and len(set(scene_ids)) == len(scene_ids)
        scene_ids = scene_ids[:num_scenes]
        descriptions = [scenes[sid] for sid in scene_ids]
        optimizer = WarpGraspOptimizer(
            cfg.grasp_optim,
            cfg.robot_config,
            device,
            fixed_base_robot_config=cfg.traj_builder.fixed_base_robot_config,
        )
        builder = build_planning_traj_builder(cfg, device, robot_config=cfg.robot_config)
        batch = GraspKitData(scene_ids=scene_ids, scene_descriptions=descriptions)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        batch.grasp_qpos = optimizer.optimize(descriptions, num_samples)
        torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - start
        start = time.perf_counter()
        batch, filter_success = builder.build(batch)
        torch.cuda.synchronize(device)
        planning_seconds = time.perf_counter() - start
        assert batch.full_traj_qpos.shape[:2] == (num_scenes, num_samples)
        saved = {
            "experiment": experiment,
            "seed": 0,
            "scene_ids": scene_ids,
            "scene_description_path": str(scene_path),
            "protocol": asdict(cfg.grasp_optim.evaluator),
            "full_traj_qpos": batch.full_traj_qpos.cpu(),
            "filter_success": filter_success.cpu(),
            "planning_success": (batch.pre_traj_penetration <= 0.005).cpu(),
        }
        payload.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, payload)
        result = {
            "stage": stage,
            "experiment": experiment,
            "seed": 0,
            "scene_ids": scene_ids,
            "trajectory_shape": list(batch.full_traj_qpos.shape),
            "optimizer_seconds": optimizer_seconds,
            "planning_seconds": planning_seconds,
            "filter_success": int(filter_success.sum()),
            "planning_success": int(saved["planning_success"].sum()),
        }
    else:
        saved = torch.load(payload, map_location="cpu", weights_only=True)
        assert saved["experiment"] == experiment and saved["seed"] == 0
        assert saved["protocol"] == asdict(cfg.grasp_optim.evaluator), "Evaluation protocol changed"
        scene_path = Path(ensure_local_filepath(str(scene_path or saved["scene_description_path"])))
        scenes = SceneDescription.load_scene_descriptions_msgpack(str(scene_path), return_splits=True)["train"]
        batch = GraspKitData(
            scene_ids=saved["scene_ids"],
            scene_descriptions=[scenes[sid] for sid in saved["scene_ids"]],
            full_traj_qpos=saved["full_traj_qpos"].to(device),
        )
        evaluator_cfg = replace(cfg.grasp_optim.evaluator, num_envs=num_envs)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        evaluator = build_env(evaluator_cfg, cfg.robot_config, device, enable_rendering=enable_rendering)
        torch.cuda.synchronize(device)
        initialization_seconds = time.perf_counter() - start
        measurements, metrics = [], []
        total_samples = batch.full_traj_qpos.shape[0] * batch.full_traj_qpos.shape[1]
        for repeat in range(repeats):
            seed_everything(0)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            infos = evaluator.evaluate(batch)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            infos = {name: values.cpu() for name, values in infos.items()}
            metrics.append(infos)
            measurements.append(
                {
                    "repeat": repeat,
                    "seconds": seconds,
                    "samples_per_second": total_samples / seconds,
                    "success_count": int(infos["success"].sum()),
                    "final_success_count": int(
                        (infos["success"] & saved["filter_success"] & saved["planning_success"]).sum()
                    ),
                }
            )
        metrics_path = output.with_suffix(".pt")
        torch.save({"scene_ids": batch.scene_ids, "metrics": metrics}, metrics_path)
        result = {
            "stage": stage,
            "experiment": experiment,
            "seed": 0,
            "scene_ids": batch.scene_ids,
            "protocol": saved["protocol"],
            "num_envs": num_envs,
            "enable_rendering": enable_rendering,
            "total_samples": total_samples,
            "trajectory_shape": list(batch.full_traj_qpos.shape),
            "initialization_seconds": initialization_seconds,
            "measurements": measurements,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "metrics_path": str(metrics_path),
        }
        evaluator.close()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    tyro.cli(main)
