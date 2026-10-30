"""Unified multi-GPU pipeline launcher; `optimize_grasps` takes a floating hand or fixed arm+hand `robot_config`.

Steps:
    sample_point_cloud / render_data -> optimize_grasps -> process_sim_data -> train

Usage:
    uv run scripts/launch.py g1_inspire_right_wbc_dex1b_v0_4_1_mini --gpus 0,1,2,3

    # Restart from a step (full clean from that step onward)
    uv run scripts/launch.py inspire_right_dex1b_v0_4_mini_tabletop --restart-from optimize_grasps
    # Append mode: keep restart step's output, only delete its marker
    uv run scripts/launch.py inspire_right_dex1b_v0_4_mini_tabletop --restart-from optimize_grasps --append
"""

import glob
import logging
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from graspkit.configs.experiment import cli_experiment
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.constants import PROJECT_ROOT
from graspkit.utils.gpu_utils import resolve_gpus

sys.path.append(str(PROJECT_ROOT))
logger = logging.getLogger("graspkit")


def cleanup_for_restart(cfg: ExperimentConfig) -> list[str]:
    """Delete markers + outputs from restart_from onward. With --append, keep restart step's output."""
    from graspkit.stages.grasp_optim.optimize_pipeline import OPTIM_COMPLETE_MARKER
    from scripts.stages.data_process.process_sim_data import PROCESS_COMPLETE_MARKER
    from scripts.stages.obs_gen.render_data import RENDER_COMPLETE_MARKER
    from scripts.stages.obs_gen.sample_point_cloud import SAMPLE_POINT_CLOUD_COMPLETE_MARKER
    from scripts.stages.train.train import TRAIN_COMPLETE_MARKER

    all_steps = list(cfg.steps)
    restart_idx = all_steps.index(cfg.restart_from)
    steps_to_run = all_steps[restart_idx:]

    # step -> list of (marker_globs, output_path, is_directory)
    step_cleanup: dict[str, list[tuple[list[str], str, bool]]] = {}
    if cfg.grasp_output_dir:
        step_cleanup["optimize_grasps"] = [
            (
                [os.path.join(cfg.grasp_output_dir, OPTIM_COMPLETE_MARKER + "*")],
                cfg.grasp_output_dir,
                True,
            )
        ]
    step_cleanup["render_data"] = [([cfg.rendered_data_path + RENDER_COMPLETE_MARKER], cfg.rendered_data_path, True)]
    if cfg.object_pointcloud_path:
        step_cleanup["sample_point_cloud"] = [
            (
                [cfg.object_pointcloud_path + SAMPLE_POINT_CLOUD_COMPLETE_MARKER],
                cfg.object_pointcloud_path,
                False,
            )
        ]
    if cfg.processed_data_path:
        step_cleanup["process_sim_data"] = [
            (
                [cfg.processed_data_path + PROCESS_COMPLETE_MARKER],
                cfg.processed_data_path,
                False,
            )
        ]
    train_dir = str(Path("logs/train") / cfg.name)
    if os.path.isdir(train_dir):
        step_cleanup["train"] = [
            (
                [os.path.join(train_dir, TRAIN_COMPLETE_MARKER)],
                train_dir,
                True,
            )
        ]

    for i, step in enumerate(steps_to_run):
        if step not in step_cleanup:
            continue
        keep_output = cfg.append and i == 0
        for marker_globs, output_path, is_dir in step_cleanup[step]:
            for pattern in marker_globs:
                for marker in glob.glob(pattern):
                    logger.info(f"Removing marker: {marker}")
                    os.remove(marker)
            if keep_output:
                continue
            if is_dir and os.path.isdir(output_path):
                logger.info(f"Removing directory: {output_path}")
                shutil.rmtree(output_path)
            elif not is_dir and os.path.isfile(output_path):
                logger.info(f"Removing file: {output_path}")
                os.remove(output_path)

    return steps_to_run


def _optimize_grasps_worker(rank: int, world_size: int, cfg: ExperimentConfig):
    from graspkit.stages.grasp_optim.optimize_pipeline import optimize_grasps

    if rank != 0:
        # Stagger launches so rank 0's Warp kernel compilation finishes before other ranks attempt it.
        time.sleep(20)
        sys.stdout = open(os.devnull, "w")
    cfg.rank = rank
    cfg.world_size = world_size
    optimize_grasps(cfg)


def _spawn_ranks(
    worker_fn, worker_args: tuple, cfg: ExperimentConfig, gpu_ids: list[int], active_processes: list, step_name: str
) -> None:
    world_size = len(gpu_ids)
    ctx = mp.get_context("spawn")
    for rank, gpu_id in enumerate(gpu_ids):
        logger.info(f"Launching {step_name} rank {rank}/{world_size} on GPU {gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = ctx.Process(target=worker_fn, args=(rank, world_size, cfg, *worker_args))
        p.start()
        active_processes.append(p)

    for p in active_processes:
        p.join()

    failed = [(rank, p.pid, p.exitcode) for rank, p in enumerate(active_processes) if p.exitcode != 0]
    assert not failed, f"{step_name} processes failed at ranks: {failed}"
    active_processes.clear()


def run_optimize_grasps(cfg: ExperimentConfig, gpu_ids: list[int], active_processes: list):
    from graspkit.stages.grasp_optim.optimize_pipeline import OPTIM_COMPLETE_MARKER

    assert cfg.grasp_output_dir, "optimize_grasps requires grasp_output_dir"
    marker = os.path.join(cfg.grasp_output_dir, OPTIM_COMPLETE_MARKER)
    if not cfg.grasp_output_dir.startswith("s3://") and os.path.exists(marker):
        logger.info(f"Grasp optimization already complete (found {OPTIM_COMPLETE_MARKER}), skipping.")
        return
    _spawn_ranks(
        _optimize_grasps_worker,
        (),
        cfg,
        gpu_ids,
        active_processes,
        "optimize_grasps",
    )


def _render_data_worker(
    rank: int, world_size: int, cfg: ExperimentConfig, scene_description_path: str, rendered_data_path: str
):
    from scripts.stages.obs_gen.render_data import render_scene

    cfg.scene_description_path = scene_description_path
    cfg.rendered_data_path = rendered_data_path
    render_scene(cfg)


def run_render_data(cfg: ExperimentConfig, gpu_ids: list[int], active_processes: list):
    from scripts.stages.obs_gen.render_data import RENDER_COMPLETE_MARKER

    marker_path = cfg.rendered_data_path + RENDER_COMPLETE_MARKER
    if os.path.exists(marker_path):
        logger.info(f"Render already complete at {cfg.rendered_data_path}, skipping.")
        return
    logger.info(f"Rendering scene {cfg.scene_description_path} -> {cfg.rendered_data_path}")
    _spawn_ranks(
        _render_data_worker,
        (cfg.scene_description_path, cfg.rendered_data_path),
        cfg,
        [gpu_ids[0]],
        active_processes,
        f"render_data [{cfg.rendered_data_path}]",
    )


def run_sample_point_cloud(cfg: ExperimentConfig):
    from scripts.stages.obs_gen.sample_point_cloud import (
        SAMPLE_POINT_CLOUD_COMPLETE_MARKER,
        sample_point_cloud,
    )
    from scripts.stages.obs_gen.sample_point_cloud import (
        Config as SamplePCConfig,
    )

    marker_path = cfg.object_pointcloud_path + SAMPLE_POINT_CLOUD_COMPLETE_MARKER
    if os.path.exists(marker_path):
        logger.info(f"Sample point cloud already complete (found {SAMPLE_POINT_CLOUD_COMPLETE_MARKER}), skipping.")
        return

    pc_cfg = SamplePCConfig(
        scene_description_path=cfg.scene_description_path,
        output_path=cfg.object_pointcloud_path,
        num_points=cfg.renderer.sample_num_points,
        enable_fps=cfg.renderer.sample_enable_fps,
    )
    sample_point_cloud(pc_cfg)


def _process_sim_data_worker(rank: int, world_size: int, cfg: ExperimentConfig):
    from scripts.stages.data_process.process_sim_data import create_dataset

    create_dataset(cfg)


def run_process_sim_data(cfg: ExperimentConfig, gpu_ids: list[int], active_processes: list):
    from scripts.stages.data_process.process_sim_data import PROCESS_COMPLETE_MARKER

    marker_path = cfg.processed_data_path + PROCESS_COMPLETE_MARKER
    if os.path.exists(marker_path):
        logger.info(f"Process sim data already complete (found {PROCESS_COMPLETE_MARKER}), skipping.")
        return

    _spawn_ranks(_process_sim_data_worker, (), cfg, [gpu_ids[0]], active_processes, "process_sim_data")


def run_train(cfg: ExperimentConfig, gpu_ids: list[int], active_processes: list):
    from scripts.stages.train.train import TRAIN_COMPLETE_MARKER

    marker_path = Path(f"logs/train/{cfg.name}") / TRAIN_COMPLETE_MARKER
    if marker_path.exists():
        logger.info(f"Training already complete (found {TRAIN_COMPLETE_MARKER}), skipping.")
        return

    if cfg.train_single_gpu:
        gpu_ids = gpu_ids[:1]
    gpu_str = ",".join(str(g) for g in gpu_ids)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str}
    if len(gpu_ids) > 1:
        cmd = [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            f"--nproc-per-node={len(gpu_ids)}",
            "scripts/stages/train/train.py",
            cfg.name,
        ]
    else:
        cmd = ["uv", "run", "scripts/stages/train/train.py", cfg.name]
    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    active_processes.append(proc)
    proc.wait()
    active_processes.clear()
    assert proc.returncode == 0, f"Training failed with return code {proc.returncode}"


def main():
    cfg: ExperimentConfig = cli_experiment()

    gpu_ids = resolve_gpus(cfg.gpus, cfg.num_gpus, cfg.gpu_memory_threshold_mb)
    logger.info(f"Using GPUs: {gpu_ids}")

    if cfg.restart_from:
        assert cfg.restart_from in cfg.steps, f"restart_from={cfg.restart_from!r} not in steps {list(cfg.steps)}"
        steps_to_run = cleanup_for_restart(cfg)
        logger.info(f"Restarting from {cfg.restart_from!r}, running steps: {steps_to_run}")
    else:
        steps_to_run = list(cfg.steps)

    active_processes: list = []

    def signal_handler(signum, frame):
        # Restore default handlers so a second Ctrl+C kills the launcher immediately.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print()  # Ensure a new line
        logger.error(f"Received signal {signum}, terminating child processes (Ctrl+C again to force)...")
        for p in active_processes[:]:
            if isinstance(p, subprocess.Popen):
                pgid = os.getpgid(p.pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    p.wait(timeout=5)
            else:
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
            active_processes.remove(p)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for step in steps_to_run:
        if step == "optimize_grasps":
            logger.info("Starting step: optimize_grasps")
            run_optimize_grasps(cfg, gpu_ids, active_processes)
            logger.info("Completed step: optimize_grasps")
        elif step == "render_data":
            logger.info("Starting step: render_data")
            run_render_data(cfg, gpu_ids, active_processes)
            logger.info("Completed step: render_data")
        elif step == "sample_point_cloud":
            logger.info("Starting step: sample_point_cloud")
            run_sample_point_cloud(cfg)
            logger.info("Completed step: sample_point_cloud")
        elif step == "process_sim_data":
            logger.info("Starting step: process_sim_data")
            run_process_sim_data(cfg, gpu_ids, active_processes)
            logger.info("Completed step: process_sim_data")
        elif step == "train":
            logger.info("Starting step: train")
            run_train(cfg, gpu_ids, active_processes)
            logger.info("Completed step: train")
        else:
            raise ValueError(f"Unknown pipeline step: {step}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
