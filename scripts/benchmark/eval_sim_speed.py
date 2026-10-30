"""Time cold/warm `evaluate()` on one grasp batch to compare ManiSkill `MsEnv` (GPU) and MuJoCo `MjEnv` (CPU) speed.

Usage:
    uv run scripts/benchmark/eval_sim_speed.py --experiment-name inspire_right_dex1b_v0_4_mini_floating --num-scenes 4 --num-samples 16
    uv run scripts/benchmark/eval_sim_speed.py --experiment-name shadow_right_mj_val --num-scenes 4 --num-samples 16
"""

import graspkit  # noqa: F401 I001

import logging
import time

import torch
import tyro

from graspkit.configs.experiment import build_experiment_config
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import MjEnv, build_env
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.seed_utils import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark(experiment_name: str, num_scenes: int = 4, num_samples: int = 16) -> None:
    seed_everything(42)
    cfg = build_experiment_config(experiment_name)
    g = cfg.grasp_optim
    robot_config = cfg.traj_builder.floating_base_robot_config
    scenes = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    descs = [scenes[sid] for sid in sorted(scenes)[:num_scenes]]

    optimizer = WarpGraspOptimizer(
        g, robot_config, device, fixed_base_robot_config=cfg.traj_builder.fixed_base_robot_config
    )
    motion_planner = build_motion_planner(cfg.traj_builder.motion_planner, robot_config=robot_config, device=device)
    # rollouts_per_grasp: per-rollout normalizer for a fair cross-protocol comparison.
    # - MuJoCo: 6 force-direction force-closure rollouts (early break on failure -> nominal max)
    # - ManiSkill: num_seeds full-trajectory rollouts (GPU-batched over all grasps at once)
    ev_cfg = cfg.grasp_optim.evaluator
    evaluator = build_env(ev_cfg, robot_config, device)
    if isinstance(evaluator, MjEnv):
        rollouts_per_grasp = 6
        protocol = "MuJoCo: 6 FC rollouts/grasp, CPU (48-worker pool)"
    else:
        rollouts_per_grasp = ev_cfg.num_seeds
        protocol = f"ManiSkill: {ev_cfg.num_seeds} traj rollouts/grasp, GPU-batched over all grasps"

    batch = GraspKitData(scene_ids=[d.scene_id for d in descs], scene_descriptions=descs)
    batch.grasp_qpos = optimizer.optimize(descs, num_samples)
    batch = motion_planner.plan(batch, return_pre_post_only=False)

    n = num_scenes * num_samples
    traj_len = batch.full_traj_qpos.shape[-2]
    warm = 0.0
    for label in ("cold", "warm"):
        t = time.time()
        infos = evaluator.evaluate(batch)
        warm = time.time() - t
        logger.info(f"[{label}] {n} grasps in {warm:.2f}s | SR={infos['success'].float().mean():.1%}")
    evaluator.close()

    # Physics-steps/s is the fair throughput unit (normalizes rollout count AND length; per-grasp differs by protocol).
    # - MuJoCo: actual mj_step calls (fewer when a grasp fails early)
    # - ManiSkill: traj_len control frames at sim/control freq 100/50 = 2 substeps, per seed
    if isinstance(evaluator, MjEnv):
        total_steps = int(infos["sim_steps"].sum())
    else:
        total_steps = n * traj_len * 2 * rollouts_per_grasp
    logger.info(
        f"=== {experiment_name} | {protocol} | warm {warm:.2f}s | {total_steps} steps -> {total_steps / warm:.0f} steps/s ==="
    )


if __name__ == "__main__":
    tyro.cli(benchmark)
