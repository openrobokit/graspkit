"""Benchmark motion planners on fixed-base presets: timing, success and smoothness of the whole start → lift plan.

Variants: `warp` optimizes start → pre-grasp and fills the other phases at their step counts; `warp_joint`
(`enable_joint_optimization`) plans start → pre-grasp → grasp → hold → lift in one robokit call; `torch` the legacy one.

Presets:
  - xarm7_xhand_right: xarm_xhand_dex1b_tabletop_mini
  - g1_inspire_right: g1_inspire_right_dex1b_mini_tabletop

Usage:
    # One-time fixture generation (run once, commit outputs):
    uv run scripts/benchmark/bench_motion_gen.py xarm7_xhand_right --setup_grasps
    uv run scripts/benchmark/bench_motion_gen.py g1_inspire_right --setup_grasps

    # Benchmark runs (load committed fixtures):
    uv run scripts/benchmark/bench_motion_gen.py xarm7_xhand_right
    uv run scripts/benchmark/bench_motion_gen.py g1_inspire_right --viz
    uv run scripts/benchmark/bench_motion_gen.py all --save_markdown
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import trimesh
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.constants import OBJECT_COLOR, PROJECT_ROOT, ROBOT_COLOR, TABLE_COLOR
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.seed_utils import seed_everything

torch.set_float32_matmul_precision("high")
logger = logging.getLogger("graspkit")
logging.basicConfig(level=logging.INFO)


Variant = Literal["torch", "warp", "warp_joint"]
DEV_VARIANTS: tuple[Variant, ...] = ("warp", "warp_joint")
FIXTURE_DIR = Path(__file__).parent / "motion_gen_grasps"


@dataclass
class BenchMotionGenConfig:
    experiment_name: str = "xarm7_xhand_right_dex1b_v0_4_mini_tabletop"
    preset_name: str = "xarm7_xhand_right"
    """Name of the fixture subdirectory under scripts/benchmark/motion_gen_grasps/."""
    split: Literal["train", "test"] = "train"
    num_scenes: int = 8
    num_samples_per_scene: int = 128
    variants: tuple[Variant, ...] = ()
    """Variants to benchmark. Empty defaults to the dev set (see code)."""
    include_torch: bool = False
    """Include `torch` in the default variant set (slow; disabled by default)."""
    setup_grasps: bool = False
    """When True, generate + save fixed-base grasp fixtures for this preset and exit."""
    save_markdown: bool = False
    run_all: bool = False
    """When True, run all presets serially and save combined markdown results."""
    seed: int = 42
    viz: bool = False
    sim_eval: bool = True
    """Roll the planned trajectories out in the experiment's sim evaluator (no perturbation) for the lift success rate."""
    num_steps_pre_grasp: Optional[int] = None
    """Override pre-grasp trajectory length for this benchmark (e.g. 150); None uses the experiment config's value."""


ALL_PRESET_NAMES: tuple[str, ...] = ("xarm7_xhand_right", "g1_inspire_right")


def _resolve_variants(cfg: BenchMotionGenConfig) -> tuple[Variant, ...]:
    if cfg.variants:
        return cfg.variants
    return ("torch",) + DEV_VARIANTS if cfg.include_torch else DEV_VARIANTS


# ---------------------------------------------------------------------------
# Fixture generation (--setup_grasps)
# ---------------------------------------------------------------------------


def generate_grasp_fixtures(cfg: BenchMotionGenConfig) -> None:
    """Run warp_gd fixed-base grasp optim, saving .h5 per scene; arm-space optimization keeps grasps IK-reachable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    exp_cfg = build_experiment_config(cfg.experiment_name)
    scene_splits = SceneDescription.load_scene_descriptions_msgpack(exp_cfg.scene_description_path, return_splits=True)
    scenes = scene_splits.get(cfg.split) or next(iter(scene_splits.values()))
    scene_ids = sorted(scenes.keys())[: cfg.num_scenes]
    scene_descriptions = [scenes[sid] for sid in scene_ids]
    print(
        f"Generating grasp fixtures for {cfg.preset_name}: "
        f"{len(scene_descriptions)} scenes × {cfg.num_samples_per_scene} grasps (fixed-base warp_gd)"
    )

    grasp_cfg = replace(
        exp_cfg.grasp_optim,
        optimizer_backend="warp_gd",
        enable_ik_scoring=False,
        num_samples_per_batch=len(scene_descriptions) * cfg.num_samples_per_scene,
    )
    optimizer = WarpGraspOptimizer(grasp_cfg, exp_cfg.traj_builder.fixed_base_robot_config, device)
    grasp_qpos = optimizer.optimize(scene_descriptions, cfg.num_samples_per_scene)
    grasp_only = GraspKitData(scene_ids=scene_ids, grasp_qpos=grasp_qpos)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURE_DIR / f"{cfg.preset_name}.h5"
    grasp_only.save(str(out_path))
    print(f"Saved {len(scene_ids)} grasp fixtures to {out_path}")


# ---------------------------------------------------------------------------
# One variant run
# ---------------------------------------------------------------------------


def _run_variant(
    variant: Variant,
    exp_cfg: ExperimentConfig,
    grasp_batch: GraspKitData,
    device: torch.device,
) -> tuple[GraspKitData, dict[str, float]]:
    """Run one variant; return the planned batch and its timing."""
    # Release the previous variant's planner buffers (the robokit L-BFGS fused Jacobian alone is ~5 GB) to avoid OOM.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    motion_cfg = replace(
        exp_cfg.traj_builder.motion_planner,
        backend="torch" if variant == "torch" else "warp",
        enable_joint_optimization=variant == "warp_joint",
    )
    planner = build_motion_planner(motion_cfg, robot_config=exp_cfg.traj_builder.fixed_base_robot_config, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = planner.plan(grasp_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, {"plan_time": time.perf_counter() - start}


def _compute_smoothness(traj: torch.Tensor, control_freq: float) -> dict[str, float]:
    """Joint acceleration and jerk at the control rate over each plan's motion, up to its last moving frame.

    RMS over joints and frames, per-plan peaks averaged, the mean motion time, and the shortest and longest hold (the
    longest still stretch before the last move); a plan's trailing padding is left out. Pass the arm joints: the hand
    is closed by the same script in every variant.
    """
    moving = torch.diff(traj, dim=-2).abs().amax(dim=-1).gt(1e-6)
    last = moving.cumsum(dim=-1).argmax(dim=-1)  # the last move
    acc = torch.diff(traj, n=2, dim=-2) * control_freq**2
    jerk = torch.diff(traj, n=3, dim=-2) * control_freq**3
    run, hold = torch.zeros_like(last), torch.zeros_like(last)
    for k in range(moving.shape[-1]):
        run = (run + 1) * (~moving[..., k] & (k < last))
        hold = torch.maximum(hold, run)
    out = {
        "motion_time": ((last + 1).float() / control_freq).mean().item(),
        "hold_min": hold.min().item() / control_freq,
        "hold_max": hold.max().item() / control_freq,
    }
    for name, value in (("acc", acc), ("jerk", jerk)):
        inside = torch.arange(value.shape[-2], device=traj.device) <= last[..., None]
        out[f"rms_{name}"] = value.pow(2).mean(dim=-1)[inside].mean().sqrt().item()
        out[f"max_{name}"] = value.abs().amax(dim=(-1, -2)).mean().item()
    return out


# ---------------------------------------------------------------------------
# Top-level bench
# ---------------------------------------------------------------------------


def run_benchmark(cfg: BenchMotionGenConfig) -> tuple[dict[str, dict], list[SceneDescription], GraspRobot]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    exp_cfg = build_experiment_config(cfg.experiment_name)
    if cfg.num_steps_pre_grasp is not None:
        exp_cfg = replace(
            exp_cfg,
            traj_builder=replace(
                exp_cfg.traj_builder,
                motion_planner=replace(
                    exp_cfg.traj_builder.motion_planner, num_steps_pre_grasp=cfg.num_steps_pre_grasp
                ),
            ),
        )
    scene_splits = SceneDescription.load_scene_descriptions_msgpack(exp_cfg.scene_description_path, return_splits=True)
    scenes = scene_splits.get(cfg.split) or next(iter(scene_splits.values()))
    scene_ids = sorted(scenes.keys())[: cfg.num_scenes]
    scene_descriptions = [scenes[sid] for sid in scene_ids]
    print(f"Loaded {len(scene_descriptions)} scenes for {cfg.preset_name}")

    fixture_path = FIXTURE_DIR / f"{cfg.preset_name}.h5"
    assert fixture_path.exists(), (
        f"Grasp fixtures missing at {fixture_path}. Run with --setup_grasps first to generate them."
    )
    loaded = GraspKitData.load(str(fixture_path), scene_ids=scene_ids, packed=False)
    assert loaded.grasp_qpos is not None, f"No grasp_qpos in fixtures at {fixture_path}"
    assert loaded.grasp_qpos.shape[1] >= cfg.num_samples_per_scene, (
        f"Fixtures have {loaded.grasp_qpos.shape[1]} grasps per scene, need {cfg.num_samples_per_scene}"
    )
    grasp_qpos = loaded.grasp_qpos[:, : cfg.num_samples_per_scene].to(device)
    print(f"Loaded grasps from fixtures: {tuple(grasp_qpos.shape)}")

    robot_helper = GraspRobot.from_config(exp_cfg.traj_builder.fixed_base_robot_config, device=device)
    evaluator_cfg = replace(
        exp_cfg.grasp_optim.evaluator,
        enable_perturbation=False,
        num_seeds=1,
        num_envs=grasp_qpos.numel() // grasp_qpos.shape[-1],
    )
    evaluator = build_env(evaluator_cfg, exp_cfg.traj_builder.fixed_base_robot_config, device) if cfg.sim_eval else None
    variants = _resolve_variants(cfg)
    print(f"Variants: {variants}")
    results: dict[str, dict] = {}
    for variant in variants:
        print(f"\n{'=' * 60}\nRunning {variant} planner...\n{'=' * 60}")
        seed_everything(cfg.seed)
        grasp_batch = GraspKitData(
            scene_ids=scene_ids, scene_descriptions=scene_descriptions, grasp_qpos=grasp_qpos.clone()
        )
        out, timing = _run_variant(variant, exp_cfg, grasp_batch, device)
        mp_success = out.pre_traj_penetration <= 0.005
        results[variant] = {
            "plan_time": timing["plan_time"],
            "traj": out.full_traj_qpos,
            "mp_sr": mp_success.float().mean().item(),
        }
        print(f"Plan time: {timing['plan_time']:.2f}s, MP SR: {results[variant]['mp_sr']:.3f}")
        if evaluator is not None:
            sim_success = mp_success & evaluator.evaluate(out)["success"].to(mp_success.device)
            results[variant]["sim_sr"] = sim_success.float().mean().item()
            print(f"Sim SR (MP and lift): {results[variant]['sim_sr']:.3f}")
        arm = out.full_traj_qpos[..., robot_helper.arm_joint_indices]
        results[variant].update(_compute_smoothness(arm, evaluator_cfg.control_freq))

    if evaluator is not None:
        evaluator.close()
    _print_table(results)
    return results, scene_descriptions, robot_helper


def _print_table(results: dict[str, dict]) -> None:
    methods = list(results.keys())
    print(f"\n{'=' * 80}\nMOTION PLANNER BENCHMARK\n{'=' * 80}")
    print(f"{'Metric':<32}", end="")
    for m in methods:
        print(f"{m.upper():<18}", end="")
    print()
    print("-" * (32 + 18 * len(methods)))
    for label, key, fmt in METRIC_ROWS:
        print(f"{label:<32}", end="")
        for m in methods:
            val = results[m].get(key)
            print(f"{val:<18{fmt}}" if val is not None else f"{'N/A':<18}", end="")
        print()
    print("=" * 80)


METRIC_ROWS = [
    ("Plan Time (s)", "plan_time", ".2f"),
    ("MP SR (pen <= 5 mm)", "mp_sr", ".3f"),
    ("Sim SR (MP and lift)", "sim_sr", ".3f"),
    ("Motion time (s)", "motion_time", ".2f"),
    ("Arm hold, min (s)", "hold_min", ".2f"),
    ("Arm hold, max (s)", "hold_max", ".2f"),
    ("Arm RMS acc (rad/s²)", "rms_acc", ".2f"),
    ("Arm RMS jerk (rad/s³)", "rms_jerk", ".1f"),
    ("Arm peak acc (rad/s²)", "max_acc", ".2f"),
    ("Arm peak jerk (rad/s³)", "max_jerk", ".1f"),
]


def _batch_info_str(cfg: BenchMotionGenConfig) -> str:
    return f"{cfg.num_scenes} scenes × {cfg.num_samples_per_scene} grasps = {cfg.num_scenes * cfg.num_samples_per_scene} trajs"


def save_results_markdown(
    all_results: dict[str, dict[str, dict]],
    batch_info: Optional[dict[str, str]] = None,
) -> str:
    """`batch_info[preset_name]` (optional) is a short string like "8 scenes × 64 grasps = 512 trajs"."""
    lines = ["# Motion Planner Benchmark Results", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    for preset_name, results in all_results.items():
        methods = list(results.keys())
        lines.append(f"\n## {preset_name}")
        if batch_info and preset_name in batch_info:
            lines.append(f"Batch: {batch_info[preset_name]}")
        lines.append("| Metric | " + " | ".join(methods) + " |")
        lines.append("|---|" + "|".join("---" for _ in methods) + "|")
        for label, key, fmt in METRIC_ROWS:
            vals = [f"{results[m].get(key):{fmt}}" if results[m].get(key) is not None else "N/A" for m in methods]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
    md_path = Path(__file__).parent / "bench_motion_gen_results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults saved to {md_path}")
    return str(md_path)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_trajectories(
    results: dict[str, dict],
    scene_descriptions: list[SceneDescription],
    robot_helper: GraspRobot,
) -> None:
    variants = list(results.keys())
    n_scenes = len(scene_descriptions)
    n_samples, n_steps = results[variants[0]]["traj"].shape[1:3]

    urdf_kwargs = {
        "build_scene_graph": True,
        "build_collision_scene_graph": True,
        "load_meshes": True,
        "load_collision_meshes": True,
    }
    if robot_helper.config.mesh_dir is not None:
        urdf_kwargs["mesh_dir"] = robot_helper.config.mesh_dir
    urdf = yourdfpy.URDF.load(robot_helper.config.urdf_or_mjcf_path, **urdf_kwargs)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    variant_dropdown = server.gui.add_dropdown(label="Variant", options=variants, initial_value=variants[0])
    scene_slider = server.gui.add_slider(label="Scene", min=0, max=max(1, n_scenes - 1), step=1, initial_value=0)
    sample_slider = server.gui.add_slider(label="Sample", min=0, max=max(1, n_samples - 1), step=1, initial_value=0)
    step_slider = server.gui.add_slider(label="Step", min=0, max=max(1, n_steps - 1), step=1, initial_value=0)

    base_pos = np.array(robot_helper.config.init_position)
    base_wxyz = np.array(robot_helper.config.init_quaternion)
    server.scene.add_frame("/robot_base", position=base_pos, wxyz=base_wxyz, axes_length=0.05, axes_radius=0.002)
    robot_handle = ViserUrdf(server, urdf, root_node_name="/robot_base/robot", mesh_color_override=ROBOT_COLOR)

    mesh_handles: dict[str, viser.MeshHandle] = {}
    current_scene = -1

    def update():
        nonlocal current_scene
        scene_idx = int(scene_slider.value)
        if scene_idx != current_scene:
            current_scene = scene_idx
            for h in mesh_handles.values():
                h.remove()
            mesh_handles.clear()
            desc = scene_descriptions[scene_idx]
            for name, md in desc.meshes.items():
                mesh_path = os.path.join(str(PROJECT_ROOT), md.collision_mesh_path)
                mesh = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
                scale = md.scales[0] if np.allclose(md.scales, md.scales[0]) else 1.0
                key = f"/scene/mesh_{name}"
                mesh_handles[key] = server.scene.add_mesh_simple(
                    key,
                    vertices=mesh.vertices * scale,
                    faces=mesh.faces,
                    color=OBJECT_COLOR if name == desc.target_actor_id else TABLE_COLOR,
                    flat_shading=True,
                )
                mesh_handles[key].position = np.array(md.position)
                mesh_handles[key].wxyz = np.array(md.quaternion)
            for name, bd in desc.boxes.items():
                box = trimesh.creation.box(bd.size)
                key = f"/scene/box_{name}"
                mesh_handles[key] = server.scene.add_mesh_simple(
                    key, vertices=box.vertices, faces=box.faces, color=TABLE_COLOR, flat_shading=True
                )
                mesh_handles[key].position = np.array(bd.position)
                mesh_handles[key].wxyz = np.array(bd.quaternion)

        qpos = results[variant_dropdown.value]["traj"][scene_idx, int(sample_slider.value), int(step_slider.value)]
        # Atomic: push all per-link frame updates in one message so the robot doesn't render half-updated (shaky).
        with server.atomic():
            robot_handle.update_cfg(qpos.cpu().numpy())

    for ctrl in (variant_dropdown, scene_slider, sample_slider, step_slider):
        ctrl.on_update(lambda _event: update())
    update()
    input("Press Enter to exit...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


CONFIGS: dict[str, BenchMotionGenConfig] = {
    "xarm7_xhand_right": BenchMotionGenConfig(
        experiment_name="xarm7_xhand_right_dex1b_v0_4_mini_tabletop",
        preset_name="xarm7_xhand_right",
    ),
    "g1_inspire_right": BenchMotionGenConfig(
        experiment_name="g1_inspire_right_dex1b_mini_tabletop",
        preset_name="g1_inspire_right",
    ),
    "all": BenchMotionGenConfig(run_all=True, save_markdown=True),
}


if __name__ == "__main__":
    config: BenchMotionGenConfig = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    if config.setup_grasps:
        generate_grasp_fixtures(config)
    elif config.run_all:
        all_results: dict[str, dict] = {}
        all_batch_info: dict[str, str] = {}
        for preset in ALL_PRESET_NAMES:
            sub = replace(
                CONFIGS[preset],
                seed=config.seed,
                save_markdown=config.save_markdown,
                include_torch=config.include_torch,
            )
            results, _, _ = run_benchmark(sub)
            all_results[preset] = {v: {k: v2 for k, v2 in r.items() if k != "traj"} for v, r in results.items()}
            all_batch_info[preset] = _batch_info_str(sub)
        if config.save_markdown:
            save_results_markdown(all_results, batch_info=all_batch_info)
    else:
        results, scene_descs, robot_helper = run_benchmark(config)
        if config.save_markdown:
            results_clean = {v: {k: v2 for k, v2 in r.items() if k != "traj"} for v, r in results.items()}
            batch_info = {config.preset_name: _batch_info_str(config)}
            save_results_markdown({config.preset_name: results_clean}, batch_info=batch_info)
        if config.viz:
            visualize_trajectories(results, scene_descs, robot_helper)
