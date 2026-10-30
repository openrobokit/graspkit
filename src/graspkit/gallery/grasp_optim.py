"""Grasp gallery: optimize hands floating and jointly on every arm in `PLACEMENTS`, cached and browsed in one viser page.

Usage:
    uv run python -m graspkit.gallery.grasp_optim
    uv run python -m graspkit.gallery.grasp_optim --hand sharpa_right --object_key ycb/power_drill --num_iters 4000
    uv run python -m graspkit.gallery.grasp_optim --experiment_name sharpa_right_dex1b_v0_5_mini_tabletop_gd
    uv run python -m graspkit.gallery.grasp_optim --all --gpus 0 3 4 6 7
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import hashlib
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import trimesh
import tyro
import viser
import yourdfpy
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import euler_angles_to_matrix as euler_angles_to_matrix_np
from robokit.xform.numpy import matrix_to_quaternion
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix
from torch.types import Device
from viser.extras import ViserUrdf

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import (
    EnvConfig,
    ExperimentConfig,
    GraspOptimConfig,
    GraspTrajBuilderConfig,
    MotionPlannerConfig,
)
from graspkit.configs.robot import build_robot_config
from graspkit.constants import DATA_ROOT, OBJECT_COLOR, ROBOT_COLOR, TABLE_COLOR
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.configs.robot_config import HandAxis
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_filters import Candidates, Ori, Pen, Sdf, compute_self_collision_depth
from graspkit.stages.grasp_optim import WarpGraspOptimizer
from graspkit.stages.grasp_traj_build import build_planning_traj_builder
from graspkit.utils.assembly_utils import PLACEMENTS, assembled_robot_config

logger = logging.getLogger("graspkit")

# Hands the gallery optimizes for. The arms each can mount come from `PLACEMENTS`, not a table repeated here.
GALLERY_HANDS: tuple[str, ...] = (
    "xhand_right",
    "ability_right",
    "sharpa_right",
    "inspire_right",
    "mano_right",
    "allegro_right",
    "dclaw_right",
    "paxini_right",
    "orca_right",
    "robotiq_2f85",
    "yam_gripper",
    "leap_right",
    "barrett_right",
)

GALLERY_EXPERIMENTS: dict[str, str] = {
    "ability_right": "ability_right_dex1b_v0_4_mini_tabletop",
    "inspire_right": "inspire_right_dex1b_v0_5_mini_tabletop",
    "mano_right": "mano_right_dex1b_v0_4_mini_tabletop",
    "paxini_right": "paxini_right_dex1b_v0_4_mini_floating",
    "sharpa_right": "sharpa_right_dex1b_v0_5_mini_tabletop_gd",
    "xhand_right": "xhand_right_dex1b_v0_5_mini_tabletop",
}
"""Experiment whose `grasp_optim` a hand uses (tabletop where one exists); others run `GraspOptimConfig()`."""

GALLERY_OBJECTS: tuple[str, ...] = (
    "ycb/mustard_bottle",
    "ycb/power_drill",
    "ycb/tomato_soup_can",
    "ycb/sugar_box",
    "ycb/rubiks_cube",
    "ycb/apple",
    "ycb/hammer",
    "ycb/pear",
)
"""dex_diffuser `<source>/<name>` meshes the gallery can place on the table."""

TABLE_SIZE: tuple[float, float, float] = (2.0, 2.0, 1.0)
"""Gallery table, top at z=0; wide and thick since nothing pushes back a hand slipping beside or under a small slab."""
ARM_TABLE_SIZE: tuple[float, float, float] = (0.5, 0.5, 1.0)
"""Arm-sim table, the pipeline's v0.5 tabletop footprint; every base would stand inside `TABLE_SIZE`."""
# 256 per plan batch: a humanoid rig's dense trajectory Jacobian over 1024 grasps asks for 32 GB
ARM_MOTION_PLANNER = MotionPlannerConfig(
    post_grasp_hand_grad_move=0.1, num_steps_pre_to_grasp=40, num_samples_per_batch=256
)
"""Arm-sim planner; at `pre_grasp_arm_grad_move > 0` the pre-grasp hand is the rest pose, not a gradient step."""
ARM_MOTION_PLANNERS: dict[str, MotionPlannerConfig] = {
    # A gripper needs the full squeeze; 0.1 left robotiq/xarm7 at 2.7% (1.0: 16-19%), xhand prefers 0.1.
    "robotiq_2f85": replace(ARM_MOTION_PLANNER, post_grasp_hand_grad_move=1.0),
    # yam's open jaws swept the object away on the way in; backing the pre-grasp off: franka 1.2% -> 41.4%.
    "yam_gripper": replace(ARM_MOTION_PLANNER, pre_grasp_arm_grad_move=0.015, post_grasp_hand_grad_move=1.0),
}
"""Per-hand overrides of `ARM_MOTION_PLANNER`, tuned on one arm's cached grasps."""
TABLE_DRAWN_THICKNESS: float = 0.05
"""Only the top of the table is drawn."""


FILTER_NAMES: tuple[str, ...] = ("sdf", "pen", "scene_collision", "self_collision", "ori")
CACHE_DIR = DATA_ROOT / "gallery"
"""Runs cached by (hand, object, params), since a hand costs ~25s; `rm -rf` it after changing what a run stores."""


def gallery_config(hand: str) -> GraspOptimConfig:
    """The hand's experiment `grasp_optim` (or the default) minus its evaluator, a flat config the cache can rebuild."""
    experiment = GALLERY_EXPERIMENTS.get(hand)
    cfg = build_experiment_config(experiment).grasp_optim if experiment else GraspOptimConfig()
    return replace(cfg, evaluator=None)


KNOBS: tuple[tuple[str, float, float, float], ...] = (
    ("num_samples_per_scene", 1, 2048, 1),
    ("num_iters", 100, 50000, 100),
    ("num_contact_points", 1, 16, 1),
    ("w_pen", 0.0, 10000.0, 10.0),
    ("w_spen", 0.0, 1000.0, 1.0),
    ("w_fingertip_dis", 0.0, 1000.0, 1.0),
    ("spen_threshold", 0.0, 0.1, 0.005),
    ("switch_possibility", 0.0, 1.0, 0.05),
    ("distance_lower", 0.0, 1.0, 0.01),
    ("distance_upper", 0.0, 1.0, 0.01),
    ("front_theta_lower", -math.pi, math.pi, 0.1),
    ("front_theta_upper", -math.pi, math.pi, 0.1),
)
"""`GraspOptimConfig` fields the page lets you tune over the hand's config: (name, min, max, step)."""


@dataclass
class GalleryRun:
    """One optimizer run: the grasps it produced plus the settings that produced them."""

    hand: str
    object_key: str
    cfg: GraspOptimConfig
    grasps: torch.Tensor
    """(num_samples, 6 + hand_dof) floating grasps: xyz, euler rxyz, hand joints."""
    seconds: float
    costs: dict[str, torch.Tensor] = field(default_factory=dict)
    """Term name -> (num_samples,) cost from `WarpGraspOptimizer.compute_costs`, saved with the grasps."""
    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    """Filter name -> (num_samples,) bool, filled by `filter_masks` and refreshed when a threshold changes."""
    arm_grasps: dict[str, torch.Tensor] = field(default_factory=dict)
    """Arm name -> (num_samples, rig_chain_dof) grasps optimized jointly on that arm+hand rig, by `optimize_arms`."""
    sim: torch.Tensor | None = None
    """(num_samples,) bool floating-hand sim success, filled by `evaluate_runs` and cached alongside the grasps."""
    arm_sim: dict[str, torch.Tensor] = field(default_factory=dict)
    """Arm name -> (num_samples,) bool sim success of `arm_grasps[arm]`."""


def gallery_scene(object_key: str, table_size: tuple[float, float, float] = TABLE_SIZE) -> SceneDescription:
    """One dex_diffuser mesh resting at the world origin on a table whose top is z=0."""
    source, name = object_key.split("/")
    model_dir = dex_diffuser.DATA_DIR / "processed" / source / name
    obj = MeshDescription(
        actor_id=name,
        collision_mesh_path=str(model_dir / "collision.glb"),
        scales=(1.0, 1.0, 1.0),
        visual_mesh_path=str(model_dir / "visual.glb"),
    )
    obj.position = (0.0, 0.0, -obj.get_bounds()[2][0])
    table = BoxDescription(actor_id="table", size=table_size, position=(0.0, 0.0, -table_size[2] / 2))
    return SceneDescription(scene_id=object_key, target_actor_id=name, boxes={"table": table}, meshes={name: obj})


def gallery_robot(robot_name: str, device: Device) -> GraspRobot:
    """Robot with meshes, plus collision spheres if shipped (some arm rigs have none); one cached instance per robot."""
    config = assembled_robot_config(robot_name)
    return GraspRobot.from_config(
        config,
        load_meshes=True,
        load_collision_spheres=config.collision_spheres_path is not None,
        device=device,
    )


def optimize_hand(
    hand: str, scenes: list[SceneDescription], cfg: GraspOptimConfig, device: Device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize all scenes' floating grasps in ONE call -> (len(scenes), num_samples, 6 + hand_dof) + per-term costs."""
    optimizer = WarpGraspOptimizer(cfg, build_robot_config(hand), device)
    grasps = optimizer.optimize(scenes, cfg.num_samples_per_scene)
    return grasps.cpu(), optimizer.compute_costs(scenes, grasps)


def fingertip_distances(run: GalleryRun) -> torch.Tensor:
    """(num_samples, num_fingertips) fingertip gaps in meters, undone from the cost: `w·|d|` (GD), `½(w·d)²` (LM)."""
    costs = torch.stack([c for n, c in run.costs.items() if n.startswith("fingertip/")], -1)
    if run.cfg.optimizer_backend == "warp_lm":
        return (2 * costs).sqrt() / run.cfg.lm_w_fingertip_dis
    return costs / run.cfg.w_fingertip_dis


def filter_masks(
    hand: str,
    scene: SceneDescription,
    grasps: torch.Tensor,
    device: Device,
    pen_threshold: float = 0.005,
    scene_pen_threshold: float = 0.005,
    self_collision_threshold: float = 0.005,
    hand_axis: HandAxis = "four_fingers_up",
    hand_axis_target: tuple[float, float, float] = (0.0, 0.0, 1.0),
    hand_axis_max_angle_deg: float = 90.0,
) -> dict[str, torch.Tensor]:
    """Per-grasp pass/fail for every gallery filter: {name: (num_samples,) bool on cpu}."""
    robot = gallery_robot(hand, device)
    variables = grasps.to(device)[None]  # (1, N, 6 + dof); the filters are written for (n_scene, n_cand, ...)
    scene_geom = SceneGeom([scene], device=device)
    cands = Candidates(
        tl=variables[..., :3],
        rot=euler_angles_to_matrix(variables[..., 3:6], "rxyz"),
        robot=robot,
        scene_geom=scene_geom,
        qpos=variables,
    )
    spheres = robot.get_collision_spheres(robot.forward_kinematics_torch(variables))  # (1, N, S, 4)
    scene_sdf, _, _ = scene_geom.query_scene_signed_distances(spheres[..., :3].reshape(1, -1, 3))
    scene_pen = (spheres[..., 3] - scene_sdf.reshape(spheres.shape[:-1])).amax(dim=-1)
    return {
        "sdf": Sdf()(cands)[0].cpu(),
        "pen": Pen(pen_threshold)(cands)[0].cpu(),
        "scene_collision": (scene_pen <= scene_pen_threshold)[0].cpu(),
        "self_collision": (compute_self_collision_depth(variables, robot) <= self_collision_threshold)[0].cpu(),
        "ori": Ori(hand_axis, hand_axis_target, hand_axis_max_angle_deg)(cands)[0].cpu(),
    }


def cache_path(hand: str, object_key: str, cfg: GraspOptimConfig) -> Path:
    """Where one (hand, object, config) run lives on disk."""
    key = hashlib.sha1(repr((hand, object_key, asdict(cfg))).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{hand}__{object_key.replace('/', '_')}__{key}.pt"


def sim_success(
    hand: str,
    object_keys: list[str],
    scenes: list[SceneDescription],
    grasps: torch.Tensor,
    device: Device,
    num_envs: int = 256,
    arm: str | None = None,
) -> torch.Tensor:
    """Motion-plan + roll out grasps in ManiSkill -> (n_obj, num_samples) success; verdicts depend on `num_envs`."""
    robot_config = build_robot_config(hand) if arm is None else assembled_robot_config(f"{arm}_{hand}")
    motion_planner = (
        MotionPlannerConfig(pre_grasp_hand_grad_move=1.0)
        if arm is None
        else ARM_MOTION_PLANNERS.get(hand, ARM_MOTION_PLANNER)
    )
    cfg = ExperimentConfig(
        name="gallery",
        robot_config=robot_config,
        traj_builder=GraspTrajBuilderConfig(floating_base_robot_config=robot_config, motion_planner=motion_planner),
    )
    batch = GraspKitData(scene_ids=list(object_keys), scene_descriptions=list(scenes))
    batch.grasp_qpos = grasps.to(device)
    batch, filter_success = build_planning_traj_builder(cfg, device).build(batch)
    evaluator = build_env(
        EnvConfig(
            load_scenes=True,
            control_freq=50,
            object_mass=0.01,
            target_lift_dist=0.08,
            num_envs=num_envs,
            enable_perturbation=False,
        ),
        robot_config,
        device,
    )
    mp_success = batch.pre_traj_penetration <= 0.005
    success = filter_success & mp_success & evaluator.evaluate(batch)["success"].to(mp_success.device)
    evaluator.close()
    return success.cpu()


def save_run(run: GalleryRun) -> None:
    """Write a run's sim verdicts and arm grasps into its cache file, beside the floating grasps."""
    path = cache_path(run.hand, run.object_key, run.cfg)
    blob = torch.load(path, weights_only=False)
    blob["sim"], blob["arm_sim"], blob["arm_grasps"] = run.sim, run.arm_sim, run.arm_grasps
    torch.save(blob, path)


def optimize_arms(runs: list[GalleryRun], arms: list[str], device: Device) -> None:
    """Re-optimize `runs` on each arm+hand rig, one call per arm; sets `run.arm_grasps[arm]` and drops its stale sim."""
    hand, keys = runs[0].hand, [run.object_key for run in runs]
    scenes = [gallery_scene(key, ARM_TABLE_SIZE) for key in keys]
    cfg = replace(runs[0].cfg, sampler_type="fixed_base")
    for arm in arms:
        start = time.time()
        rig = assembled_robot_config(f"{arm}_{hand}")
        grasps = WarpGraspOptimizer(cfg, rig, device).optimize(scenes, cfg.num_samples_per_scene).cpu()
        for i, run in enumerate(runs):
            run.arm_grasps[arm] = grasps[i]
            run.arm_sim.pop(arm, None)
        logger.info(f"gallery: {hand} on {arm}, {len(runs)} objects optimized jointly in {time.time() - start:.1f}s")
    for run in runs:
        save_run(run)


def evaluate_runs(
    runs: list[GalleryRun],
    scenes: list[SceneDescription],
    device: Device,
    num_envs: int = 256,
    arm: str | None = None,
) -> None:
    """Fill and save `run.sim` (or `run.arm_sim[arm]`, `scenes` then unused), all objects in ONE eval batch."""
    lengths = {len(run.grasps) for run in runs}
    assert len(lengths) == 1, f"sim eval needs one num_samples across objects, got {sorted(lengths)}"
    hand, keys = runs[0].hand, [run.object_key for run in runs]
    if arm is None:
        grasps = torch.stack([run.grasps for run in runs])
    else:
        scenes = [gallery_scene(key, ARM_TABLE_SIZE) for key in keys]
        grasps = torch.stack([run.arm_grasps[arm] for run in runs])
    success = sim_success(hand, keys, scenes, grasps, device, num_envs, arm)
    for i, run in enumerate(runs):
        if arm is None:
            run.sim = success[i]
        else:
            run.arm_sim[arm] = success[i]
        save_run(run)
    logger.info(
        f"gallery: sim eval {hand} on {arm or 'floating'}, {len(runs)} objects -> {success.float().mean():.1%} overall"
    )


def cached_optimize(
    hand: str,
    object_keys: list[str],
    scenes: list[SceneDescription],
    cfg: GraspOptimConfig,
    device: Device,
    overwrite: bool = False,
) -> list[GalleryRun]:
    """Optimize uncached objects in one batch (`seconds` is that batch's time), reload the rest from `CACHE_DIR`."""
    paths = {key: cache_path(hand, key, cfg) for key in object_keys}
    runs = {key: load_cached_run(path) for key, path in paths.items() if path.exists() and not overwrite}
    todo = [key for key in object_keys if key not in runs]
    if todo:
        scene_by_key = dict(zip(object_keys, scenes))
        start = time.time()
        grasps, costs = optimize_hand(hand, [scene_by_key[key] for key in todo], cfg, device)
        seconds = time.time() - start
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, key in enumerate(todo):
            torch.save(
                {
                    "hand": hand,
                    "object_key": key,
                    "grasps": grasps[i],
                    "seconds": seconds,
                    "costs": {name: cost[i] for name, cost in costs.items()},
                    "cfg": asdict(cfg),
                },
                paths[key],
            )
            runs[key] = GalleryRun(hand, key, cfg, grasps[i], seconds, {n: c[i] for n, c in costs.items()})
        logger.info(
            f"gallery: {hand} on {len(todo)} object(s) -> {grasps.shape[1]} grasps each "
            f"in {seconds:.1f}s (one batch of {grasps.shape[0] * grasps.shape[1]} grasps)"
        )
    return [runs[key] for key in object_keys]


def load_cached_run(path: Path) -> GalleryRun:
    """Rebuild a run (grasps + the exact config that made them) from one cache file."""
    blob = torch.load(path, weights_only=False)
    return GalleryRun(
        blob["hand"],
        blob["object_key"],
        GraspOptimConfig(**blob["cfg"]),
        blob["grasps"],
        float(blob["seconds"]),
        blob["costs"],
        arm_grasps=blob.get("arm_grasps", {}),
        sim=blob.get("sim"),
        arm_sim=blob.get("arm_sim", {}),
    )


FLOATING = "floating (hand only)"
"""Arm-dropdown entry that shows the floating-hand optimization instead of an arm's joint one."""


class GalleryViewer:
    """One viser server over a `(hand, object) -> [GalleryRun]` store; every GUI change re-derives what is drawn."""

    def __init__(
        self,
        device: Device,
        port: int = 8080,
        hand: str = GALLERY_HANDS[0],
        object_key: str = GALLERY_OBJECTS[0],
    ):
        self.device = device
        self.runs: dict[tuple[str, str], list[GalleryRun]] = {}
        self.scenes: dict[str, SceneDescription] = {}
        self.robots: dict[str, tuple[viser.FrameHandle, ViserUrdf, list[int], np.ndarray]] = {}
        self.lock = threading.RLock()  # warp/CUDA calls from viser callback threads must not overlap
        self.drawn_object: tuple[str, bool] = ("", False)
        self.shown: tuple[GalleryRun | None, list[int]] = (None, [])  # run + indices passing the filters

        self.server = viser.ViserServer(port=port)
        self.server.scene.configure_default_lights(cast_shadow=False)
        gui = self.server.gui
        self.hand_gui = gui.add_dropdown("Hand", GALLERY_HANDS, initial_value=hand)
        self.arm_gui = gui.add_dropdown("Arm", (FLOATING,) + tuple(PLACEMENTS), initial_value=FLOATING)
        self.object_gui = gui.add_dropdown("Object", GALLERY_OBJECTS, initial_value=object_key)
        self.run_gui = gui.add_dropdown("Run", ("-",), initial_value="-")
        self.grasp_gui = gui.add_slider("Grasp", min=0, max=1, step=1, initial_value=0)
        self.optimize_button = gui.add_button("Optimize")
        self.sim_button = gui.add_button("Sim validation")
        self.status_gui = gui.add_markdown("")

        with gui.add_folder("Cost", expand_by_default=False):
            self.cost_gui = gui.add_markdown("")

        with gui.add_folder("Filter pass rate", expand_by_default=False):
            self.pass_gui = gui.add_markdown("")

        with gui.add_folder("Optimizer", expand_by_default=False):
            self.experiment_gui = gui.add_markdown("")
            self.backend_gui = gui.add_dropdown("optimizer_backend", ("warp_gd", "warp_lm"), initial_value="warp_gd")
            self.knob_gui = {
                name: gui.add_number(name, type(lo)(0), min=lo, max=hi, step=step) for name, lo, hi, step in KNOBS
            }
            self.reset_button = gui.add_button("Reset to config")
        self.knob_hand = ""  # the hand whose config the knobs were last reset to

        with gui.add_folder("Sim params", expand_by_default=False):
            self.sim_envs_gui = gui.add_number("num_envs", 256, min=16, max=2048, step=64)
            """Parallel sim envs: the whole gallery is one eval, so this trades VRAM for wall time."""

        with gui.add_folder("Filter params", expand_by_default=False):
            self.filter_gui = {name: gui.add_checkbox(name, initial_value=False) for name in FILTER_NAMES}
            self.sim_gui = gui.add_checkbox("sim", initial_value=False)
            """Keeps only grasps the sim lifted; the mask arrives with *Sim validation*, not with a threshold."""
            self.tips_gui = gui.add_checkbox("tips", initial_value=False)
            """Keeps grasps with `tips_count` fingertips within `tips_mm` of the object, from the fingertip costs."""
            self.pen_gui = gui.add_number("pen_threshold (m)", 0.005, min=0.0, step=0.001)
            self.scene_pen_gui = gui.add_number("scene_pen_threshold (m)", 0.005, min=0.0, step=0.001)
            self.self_pen_gui = gui.add_number("self_collision_threshold (m)", 0.005, min=0.0, step=0.001)
            self.hand_axis_gui = gui.add_dropdown("hand_axis", get_args(HandAxis), initial_value="four_fingers_up")
            self.axis_gui = gui.add_vector3("hand_axis_target", (0.0, 0.0, 1.0), step=0.1)
            self.angle_gui = gui.add_number("hand_axis_max_angle_deg", 90.0, min=0.0, max=180.0, step=5.0)
            self.tips_count_gui = gui.add_number("tips_count", 3, min=1, step=1)
            self.tips_mm_gui = gui.add_number("tips_mm", 3.0, min=0.0, step=0.5)

        # Swap the rig the moment the picker changes.
        for handle in (self.hand_gui, self.arm_gui):
            handle.on_update(lambda _: self.show_robot())
        for handle in (
            self.hand_gui,
            self.arm_gui,
            self.object_gui,
            self.run_gui,
            self.sim_gui,
            self.tips_gui,
            *self.filter_gui.values(),
        ):
            handle.on_update(lambda _: self.refresh())
        for handle in (self.pen_gui, self.scene_pen_gui, self.self_pen_gui, self.hand_axis_gui, self.axis_gui):
            handle.on_update(lambda _: self.refresh())
        for handle in (self.angle_gui, self.tips_count_gui, self.tips_mm_gui):
            handle.on_update(lambda _: self.refresh())
        self.grasp_gui.on_update(lambda _: self.show_grasp())
        self.optimize_button.on_click(lambda _: self.optimize())
        self.sim_button.on_click(lambda _: self.evaluate_sim())
        self.reset_button.on_click(lambda _: self.apply_config(gallery_config(self.hand_gui.value)))

    # ── data ────────────────────────────────────────────────────────────────
    def scene(self, object_key: str, arm: bool = False) -> SceneDescription:
        """The floating hands' wide table, or with `arm` the `ARM_TABLE_SIZE` one the rig grasps were made on."""
        if (object_key, arm) not in self.scenes:
            self.scenes[object_key, arm] = gallery_scene(object_key, ARM_TABLE_SIZE if arm else TABLE_SIZE)
        return self.scenes[object_key, arm]

    def add_run(self, run: GalleryRun) -> None:
        """Store a run; it becomes the selected one if it belongs to the hand/object currently on screen."""
        self.runs.setdefault((run.hand, run.object_key), []).append(run)
        if (run.hand, run.object_key) == (self.hand_gui.value, self.object_gui.value):
            self.refresh(select_last=True)

    def config(self) -> GraspOptimConfig:
        """The current hand's config with the page's knobs over it."""
        base = gallery_config(self.hand_gui.value)
        knobs = {name: type(getattr(base, name))(handle.value) for name, handle in self.knob_gui.items()}
        return replace(base, optimizer_backend=self.backend_gui.value, **knobs)

    def apply_config(self, cfg: GraspOptimConfig) -> None:
        self.backend_gui.value = cfg.optimizer_backend
        for name, handle in self.knob_gui.items():
            handle.value = type(handle.value)(getattr(cfg, name))

    def optimize(self) -> None:
        """Freshly optimize every gallery object for this hand in ONE call, then jointly on every arm in `PLACEMENTS`."""
        hand = self.hand_gui.value
        keys = list(GALLERY_OBJECTS)
        cfg = self.config()
        with self.lock:
            self.optimize_button.disabled = True
            self.status_gui.content = (
                f"**optimizing** {hand} on {len(keys)} objects in one batch, {cfg.num_iters} iters..."
            )
            self.cost_gui.content = self.pass_gui.content = "optimizing..."  # the stale run's numbers must not show
            runs = cached_optimize(hand, keys, [self.scene(key) for key in keys], cfg, self.device, overwrite=True)
            for run in runs:
                self.add_run(run)
            for i, arm in enumerate(PLACEMENTS):
                self.status_gui.content = f"**optimizing** {hand} jointly on {arm} ({i + 1}/{len(PLACEMENTS)})..."
                optimize_arms(runs, [arm], self.device)
            self.optimize_button.disabled = False
        self.refresh()

    def evaluate_sim(self) -> None:
        """Sim-validate every object's run with the shown run's params in ONE batch, on the picked arm or floating."""
        shown, hand, arm = self.shown[0], self.hand_gui.value, self.arm_gui.value
        if shown is None:
            self.status_gui.content = "Nothing to validate: optimize a hand first."
            return
        groups = [[r for r in self.runs.get((hand, key), []) if r.cfg == shown.cfg] for key in GALLERY_OBJECTS]
        keys = [key for key, group in zip(GALLERY_OBJECTS, groups) if group]
        if arm != FLOATING and any(arm not in group[-1].arm_grasps for group in groups if group):
            self.status_gui.content = f"Nothing to validate on {arm}: this run predates it, press *Optimize* again."
            return
        with self.lock:
            self.sim_button.disabled = True
            self.status_gui.content = f"**evaluating in sim** {hand} ({arm}) on {len(keys)} objects in one batch..."
            self.pass_gui.content = "evaluating..."
            evaluate_runs(
                [group[-1] for group in groups if group],
                [self.scene(key) for key in keys],
                self.device,
                num_envs=int(self.sim_envs_gui.value),
                arm=None if arm == FLOATING else arm,
            )
            self.sim_button.disabled = False
            self.sim_gui.value = True  # land straight on the grasps sim lifted, which is the point of evaluating
        self.refresh()

    def load_cache(self) -> None:
        """Offer every run ever cached, oldest first. Pure disk reads — nothing is optimized until Optimize."""
        paths = sorted(CACHE_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime) if CACHE_DIR.is_dir() else []
        for path in paths:
            run = load_cached_run(path)
            self.runs.setdefault((run.hand, run.object_key), []).append(run)
        logger.info(f"gallery: {len(paths)} cached runs over {len(self.runs)} hand/object pairs")

    # ── drawing ─────────────────────────────────────────────────────────────
    def robot(self, robot_name: str) -> tuple[viser.FrameHandle, object, list[int], np.ndarray]:
        """Load a robot once (URDF or MJCF), plus its chain->full joint indices; later switches toggle visibility."""
        if robot_name not in self.robots:
            config = assembled_robot_config(robot_name)
            full_robot = GraspRobot.from_config(config, prune=False)
            frame = self.server.scene.add_frame(f"/robots/{robot_name}", show_axes=False)
            if config.urdf_or_mjcf_path.endswith(".xml"):
                urdf = ViserMjcf(self.server, config.urdf_or_mjcf_path, root_node_name=f"/robots/{robot_name}")
            else:
                urdf = ViserUrdf(
                    self.server,
                    yourdfpy.URDF.load(config.urdf_or_mjcf_path, build_scene_graph=True, load_meshes=True),
                    root_node_name=f"/robots/{robot_name}",
                    mesh_color_override=ROBOT_COLOR,
                )
            chain_names = gallery_robot(robot_name, self.device).actuated_joint_names
            indices = [full_robot.actuated_joint_names.index(name) for name in chain_names]
            self.robots[robot_name] = (frame, urdf, indices, full_robot.init_qpos.cpu().numpy().astype(np.float64))
        return self.robots[robot_name]

    def draw_scene(self, object_key: str, arm: bool) -> None:
        scene = self.scene(object_key, arm)
        self.server.scene.remove_by_name("/scene")
        for name, mesh_desc in scene.meshes.items():
            mesh = trimesh.load(mesh_desc.get_visual_mesh_path(), force="mesh", process=False)
            self.server.scene.add_mesh_simple(
                f"/scene/{name}",
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                color=OBJECT_COLOR,
                flat_shading=False,
                scale=mesh_desc.scales[0],
                position=np.array(mesh_desc.position),
                wxyz=np.array(mesh_desc.quaternion),
                cast_shadow=False,
                receive_shadow=False,
            )
        for name, box in scene.boxes.items():
            self.server.scene.add_box(
                f"/scene/{name}",
                color=TABLE_COLOR,
                dimensions=(box.size[0], box.size[1], TABLE_DRAWN_THICKNESS),
                position=np.array(box.position) + [0.0, 0.0, (box.size[2] - TABLE_DRAWN_THICKNESS) / 2],
                wxyz=np.array(box.quaternion),
                cast_shadow=False,
                receive_shadow=False,
            )
        self.drawn_object = (object_key, arm)

    def show_robot(self) -> tuple[viser.FrameHandle, ViserUrdf, list[int], np.ndarray]:
        """Draw the hand or arm the dropdowns select, at rest. Independent of whether any grasp exists."""
        with self.lock:
            hand, arm = self.hand_gui.value, self.arm_gui.value
            name = hand if arm == FLOATING else f"{arm}_{hand}"
            frame, urdf, indices, rest = self.robot(name)
            for other_name, (other, _, _, _) in self.robots.items():
                other.visible = other_name == name
            config = assembled_robot_config(name)
            frame.position = np.array(config.init_position)
            frame.wxyz = np.array(config.init_quaternion)
            urdf.update_cfg(rest.copy())
            return frame, urdf, indices, rest

    def show_grasp(self) -> None:
        """Pose the selected robot at the grasp the slider points to; cheap enough to run on every slider tick."""
        with self.lock:
            frame, urdf, indices, rest = self.show_robot()
            run, kept = self.shown
            if run is None or len(kept) == 0:
                return
            index = kept[min(int(self.grasp_gui.value), len(kept) - 1)]
            arm = self.arm_gui.value
            qpos = rest.copy()
            if arm != FLOATING:
                self.cost_gui.content = f"optimized jointly on {arm}; per-term costs are kept for floating runs only"
                qpos[indices] = run.arm_grasps[arm][index].numpy()
                urdf.update_cfg(qpos)
                return
            tips = [n for n in run.costs if n.startswith("fingertip/")]
            mm = dict(zip(tips, fingertip_distances(run)[index] * 1000)) if tips else {}
            self.cost_gui.content = (
                f"{len(run.grasps)} grasps · {run.cfg.num_iters} iters · {run.seconds:.1f}s\n\n"
                "| term | cost | |\n|---|---|---|\n"
                + "\n".join(
                    f"| {n} | {c[index]:.3g} | {f'{mm[n]:.1f} mm' if n in mm else ''} |" for n, c in run.costs.items()
                )
            )
            with self.server.atomic():
                grasp = run.grasps[index].numpy()
                frame.position = grasp[:3]
                frame.wxyz = matrix_to_quaternion(euler_angles_to_matrix_np(grasp[3:6], "rxyz"))
                qpos[indices] = grasp[6:]
                urdf.update_cfg(qpos)

    def refresh(self, select_last: bool = False) -> None:
        """Re-derive everything the GUI selects: run list, filter masks, then redraw."""
        with self.lock:
            hand, arm, object_key = self.hand_gui.value, self.arm_gui.value, self.object_gui.value
            self.experiment_gui.content = f"config: `{GALLERY_EXPERIMENTS.get(hand, 'GraspOptimConfig()')}`"
            if hand != self.knob_hand:  # a new hand brings its own config; the knobs start from it
                self.apply_config(gallery_config(hand))
                self.knob_hand = hand
            if tuple(self.arm_gui.options) != (FLOATING,) + tuple(PLACEMENTS):
                self.arm_gui.options = (FLOATING,) + tuple(PLACEMENTS)  # resets the value, so re-read it
                arm = self.arm_gui.value
            if (object_key, arm != FLOATING) != self.drawn_object:
                self.draw_scene(object_key, arm != FLOATING)

            runs = self.runs.get((hand, object_key), [])
            if not runs:
                self.shown = (None, [])
                self.show_robot()  # nothing to pose it at, but the picked rig is still what to look at
                self.run_gui.options = ("-",)
                self.status_gui.content = (
                    f"**{hand}** on **{object_key}**: nothing optimized yet.\n\n"
                    "Set the knobs under *Optimizer*, then press **Optimize**."
                )
                return
            labels = [
                f"#{i + 1} {r.cfg.optimizer_backend[5:]} it={r.cfg.num_iters} n={len(r.grasps)}"
                for i, r in enumerate(runs)
            ]
            previous = self.run_gui.value
            if tuple(self.run_gui.options) != tuple(labels):
                self.run_gui.options = labels  # assigning resets the value to the first entry, so re-set it
                self.run_gui.value = previous if previous in labels and not select_last else labels[-1]
            elif select_last:
                self.run_gui.value = labels[-1]
            run = runs[labels.index(self.run_gui.value)]

            if arm != FLOATING:  # the rig grasps are their own set: the floating filters say nothing about them
                sim = run.arm_sim.get(arm)
                count = len(run.arm_grasps[arm]) if arm in run.arm_grasps else 0
                keep = torch.ones(count, dtype=torch.bool) if sim is None or not self.sim_gui.value else sim
                kept = torch.nonzero(keep).flatten().tolist()
                self.shown = (run, kept)
                self.grasp_gui.max = max(1, len(kept) - 1)
                self.grasp_gui.value = min(int(self.grasp_gui.value), self.grasp_gui.max)
                self.pass_gui.content = (
                    "" if sim is None else f"| filter | pass |\n|---|---|\n| sim | {sim.float().mean():.0%} |"
                )
                self.status_gui.content = (
                    f"not optimized on {arm}: press *Optimize* to optimize this hand on every arm"
                    if arm not in run.arm_grasps
                    else f"**{len(kept)} / {count}** shown, optimized jointly on {arm}"
                    + ("\n\npress *Sim validation* to score these" if sim is None else "")
                )
                self.show_grasp()
                return
            scene = self.scene(object_key)
            run.masks = filter_masks(
                hand,
                scene,
                run.grasps,
                self.device,
                pen_threshold=self.pen_gui.value,
                scene_pen_threshold=self.scene_pen_gui.value,
                self_collision_threshold=self.self_pen_gui.value,
                hand_axis=self.hand_axis_gui.value,
                hand_axis_target=tuple(self.axis_gui.value),
                hand_axis_max_angle_deg=self.angle_gui.value,
            )
            masks = dict(run.masks)
            active = {name: handle.value for name, handle in self.filter_gui.items()}
            sim = run.sim
            if sim is not None:
                masks["sim"] = sim
                active["sim"] = self.sim_gui.value
            if any(n.startswith("fingertip/") for n in run.costs):
                within = fingertip_distances(run) * 1000 <= self.tips_mm_gui.value
                masks["tips"] = within.sum(-1) >= int(self.tips_count_gui.value)
                active["tips"] = self.tips_gui.value

            keep = torch.ones(len(run.grasps), dtype=torch.bool)
            for name, mask in masks.items():
                if active[name]:
                    keep &= mask
            kept = torch.nonzero(keep).flatten().tolist()
            self.shown = (run, kept)
            self.grasp_gui.max = max(1, len(kept) - 1)
            self.grasp_gui.value = min(int(self.grasp_gui.value), self.grasp_gui.max)

            self.pass_gui.content = "| filter | pass |\n|---|---|\n" + "\n".join(
                f"| {'**' + n + '**' if active[n] else n} | {m.float().mean():.0%} |" for n, m in masks.items()
            )
            # The buttons already grey out while busy; this says which one to press next, which greying alone cannot.
            todo = "" if sim is not None else "\n\npress *Sim validation* to score these"
            self.status_gui.content = f"**{len(kept)} / {len(run.grasps)}** shown{todo}"
        self.show_grasp()


def headless_config(hand: str, num_samples: int | None, num_iters: int | None) -> GraspOptimConfig:
    """The hand's config with the CLI's sample/iteration overrides, which is also the cache key the table reads."""
    cfg = gallery_config(hand)
    if num_samples is not None:
        cfg = replace(cfg, num_samples_per_scene=num_samples)
    if num_iters is not None:
        cfg = replace(cfg, num_iters=num_iters)
    return cfg


def sim_table(num_samples: int | None, num_iters: int | None) -> str:
    """Markdown hand x {floating, every arm in `PLACEMENTS`} sim success over every gallery object, read from the cache."""
    rows = ["| hand | floating | " + " | ".join(PLACEMENTS) + " |", "|---|---|" + "---|" * len(PLACEMENTS)]
    for hand in GALLERY_HANDS:
        cfg = headless_config(hand, num_samples, num_iters)
        paths = [cache_path(hand, key, cfg) for key in GALLERY_OBJECTS]
        runs = [load_cached_run(path) for path in paths if path.exists()]
        cells = []
        for arm in (None, *PLACEMENTS):
            sims = [run.sim if arm is None else run.arm_sim.get(arm) for run in runs]
            cells.append(f"{torch.cat(sims).float().mean():.1%}" if runs and None not in sims else "-")
        rows.append(f"| {hand} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(
    hand: str = GALLERY_HANDS[0],
    object_key: str = GALLERY_OBJECTS[0],
    experiment_name: str | None = None,
    num_samples: int | None = None,
    num_iters: int | None = None,
    port: int = 8080,
    headless: bool = False,
    sim: bool = False,
    all: bool = False,
    gpus: tuple[int, ...] = (),
):
    """`--headless`: fill `hand`'s cache (floating, every arm) and exit; `--all`: every hand, one process per GPU."""
    if all:
        gpus = gpus or tuple(range(torch.cuda.device_count()))
        free: queue.Queue[int] = queue.Queue()
        for gpu in gpus:
            free.put(gpu)
        (CACHE_DIR / "logs").mkdir(parents=True, exist_ok=True)
        sizes = [f"--num_samples={num_samples}"] * (num_samples is not None) + [f"--num_iters={num_iters}"] * (
            num_iters is not None
        )

        def run_hand(name: str) -> int:
            gpu = free.get()
            logger.info(f"gallery: {name} on GPU {gpu}")
            with open(CACHE_DIR / "logs" / f"{name}.log", "w") as log:
                code = subprocess.run(
                    [sys.executable, "-m", "graspkit.gallery.grasp_optim", "--headless", "--sim", f"--hand={name}"]
                    + sizes,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            free.put(gpu)
            logger.info(f"gallery: {name} exited {code}")
            return code

        with ThreadPoolExecutor(len(gpus)) as pool:
            codes = dict(zip(GALLERY_HANDS, pool.map(run_hand, GALLERY_HANDS)))
        failed = [name for name, code in codes.items() if code != 0]
        print(sim_table(num_samples, num_iters))
        print(f"failed: {failed} (see {CACHE_DIR / 'logs'})" if failed else "all hands finished")
        return
    if experiment_name is not None:  # that experiment's grasp_optim becomes what Optimize runs for its hand
        hand = build_experiment_config(experiment_name).robot_config.name
        assert hand in GALLERY_HANDS, f"{experiment_name} optimizes {hand}, which the gallery does not carry"
        GALLERY_EXPERIMENTS[hand] = experiment_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if headless:
        scenes = [gallery_scene(key) for key in GALLERY_OBJECTS]
        runs = cached_optimize(
            hand, list(GALLERY_OBJECTS), scenes, headless_config(hand, num_samples, num_iters), device
        )
        optimize_arms(runs, [arm for arm in PLACEMENTS if arm not in runs[0].arm_grasps], device)
        if sim:
            for arm in (None, *PLACEMENTS):
                if (runs[0].sim if arm is None else runs[0].arm_sim.get(arm)) is None:
                    evaluate_runs(runs, scenes, device, arm=arm)
        return
    viewer = GalleryViewer(device, port=port, hand=hand, object_key=object_key)
    viewer.load_cache()
    viewer.refresh()
    if num_samples is not None:
        viewer.knob_gui["num_samples_per_scene"].value = num_samples
    if num_iters is not None:
        viewer.knob_gui["num_iters"].value = num_iters
    # viser bumps the port when the requested one is taken, so report the one it actually bound.
    logger.info(f"gallery: serving on port {viewer.server.get_port()} — press 'Optimize' to optimize a hand")
    input("Gallery serving. Press Enter to exit...")


if __name__ == "__main__":
    tyro.cli(main, use_underscores=True)
