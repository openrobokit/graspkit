"""Bench helpers: robot/contact setup, scene-geom backends, warp wrappers, and graspkit motion-plan/sim bridges."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import msgspec
import numpy as np
import torch
import trimesh
import warp as wp
from robokit.assets.robots.assembly import g1_inspire as g1_inspire_assets
from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_assets
from robokit.assets.robots.end_effectors import inspire_hand as inspire_hand_assets
from robokit.assets.robots.end_effectors import mano_hand as mano_hand_assets
from robokit.assets.robots.end_effectors import xhand as xhand_assets
from robokit.geom.geoms import BoxGeom, MeshGeom, PlaneGeom, VolumeGeom
from robokit.geom.scene import WarpScene
from robokit.geom.sdf_volume import SdfVolume

from graspkit.constants import PROJECT_ROOT
from graspkit.stages.grasp_optim import (
    GraspRobot,
    RobotConfig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GRASPKIT_DATA_ROOT = Path(os.path.expanduser("~/.graspkit/data"))
SCENE_MSGPACK_DIR = GRASPKIT_DATA_ROOT / "scene_description/dex1b/v0.4"


def resolve_mesh_path(rel_path: str) -> str:
    """Resolve a graspkit-relative mesh path (``data/object_data/...``) to an absolute repo-root path; idempotent."""
    if os.path.isabs(rel_path):
        return rel_path
    assert rel_path.startswith("data/object_data/"), f"unexpected mesh path layout: {rel_path}"
    return str(PROJECT_ROOT / rel_path)


# ---------------------------------------------------------------------------
# Scene description structs, re-exported from graspkit so this bench shares its canonical schema and msgpack format.
# ---------------------------------------------------------------------------

from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.xform.numpy import pose7_to_tf_mat

from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription  # noqa: E402


def load_scenes(
    msgpack_filename: str,
    split: Literal["train", "test"] = "train",
    num_scenes: int = 8,
) -> List[SceneDescription]:
    """Load the first `num_scenes` scenes (sorted by id) from a graspkit dev msgpack."""
    msgpack_path = SCENE_MSGPACK_DIR / msgpack_filename
    with open(msgpack_path, "rb") as f:
        data = msgspec.msgpack.decode(f.read(), type=Dict[Literal["train", "test"], Dict[str, SceneDescription]])
    scenes = data[split] or data["train"] or data["test"]
    scene_ids = sorted(scenes.keys())[:num_scenes]
    return [scenes[i] for i in scene_ids]


# ---------------------------------------------------------------------------
# Robot configs (robokit RobotConfig per preset) + bench-only optim knobs
# ---------------------------------------------------------------------------


@dataclass
class BenchGraspOptimConfig:
    """Per-preset optim knobs outside `RobotConfig` (mirrors `GraspOptimConfig`); frozen as the bench reference."""

    optim_iters: int = 6000
    """graspkit `num_iters`: GD optim iterations."""
    optim_spen_threshold: float = 0.02
    """graspkit `spen_threshold`: self-penetration margin (grasp optimizer config + energy)."""
    init_distance_upper: float = 0.2
    """graspkit `distance_upper`: upper bound on init translation distance from object surface."""
    init_hand_qpos_mu: Optional[Tuple[float, ...]] = None
    """graspkit `hand_qpos_mu`: mean hand qpos for floating-base init (None -> robot's `init_qpos`)."""


def make_robot_configs() -> Dict[str, RobotConfig]:
    mano_right = RobotConfig(
        name="mano_right",
        urdf_or_mjcf_path=str(mano_hand_assets.URDF_PATH_RIGHT),
        mesh_dir=str(mano_hand_assets.URDF_PATH_RIGHT.parent),
        collision_spheres_path=str(mano_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
        self_collision_ignore_path=str(mano_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
        base_type="floating",
        init_qpos=tuple([0.0] * 16),
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z"),
        fingertip_link_names=["link_15", "link_3", "link_6", "link_12", "link_9"],
        thumb_link_names=["link_13", "link_14", "link_15"],
        middle_link_names=["link_4", "link_5", "link_6"],
        contact_sampling_qpos=(0.0,) * 13 + (0.65, 0.0, 0.0),
    )
    xhand_right = RobotConfig(
        name="xhand_right",
        urdf_or_mjcf_path=str(xhand_assets.URDF_GLB_PATH_RIGHT),
        contact_points_path=str(xhand_assets.CONTACT_POINTS_PATH_RIGHT),
        collision_spheres_path=str(xhand_assets.COLLISION_SPHERE_PATH_RIGHT),
        self_collision_ignore_path=str(xhand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
        base_type="floating",
        init_qpos=tuple([0.1] * 12),
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z"),
        fingertip_link_names=[
            "right_hand_thumb_rota_link2",
            "right_hand_index_rota_link2",
            "right_hand_mid_link2",
            "right_hand_ring_link2",
            "right_hand_pinky_link2",
        ],
    )
    xarm7_xhand_right = RobotConfig(
        name="xarm7_xhand_right",
        urdf_or_mjcf_path=str(xarm7_xhand_assets.URDF_PATH),
        contact_points_path=str(xarm7_xhand_assets.CONTACT_POINTS_PATH),
        collision_spheres_path=str(xarm7_xhand_assets.COLLISION_SPHERE_PATH),
        self_collision_ignore_path=str(xarm7_xhand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
        base_type="fixed",
        init_position=(-0.41, -0.37, 0.01),
        init_qpos=(-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00) + (0.1,) * 12,
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="y", four_fingers_up_axis="-z"),
        arm_base_link_name="link_base",
        hand_base_link_name="right_hand_link",
        fingertip_link_names=[
            "right_hand_thumb_rota_link2",
            "right_hand_index_rota_link2",
            "right_hand_mid_link2",
            "right_hand_ring_link2",
            "right_hand_pinky_link2",
        ],
        workspace_origin=(-0.41, -0.37, 0.01),
    )
    inspire_hand_right = RobotConfig(
        name="inspire_hand_right",
        urdf_or_mjcf_path=str(inspire_hand_assets.URDF_PATH_RIGHT),
        contact_points_path=str(inspire_hand_assets.CONTACT_POINTS_PATH_RIGHT),
        collision_spheres_path=str(inspire_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
        self_collision_ignore_path=str(inspire_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
        base_type="floating",
        init_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="x", four_fingers_up_axis="z"),
        thumb_link_names=["thumb_intermediate", "thumb_proximal", "thumb_distal"],
        middle_link_names=["middle_proximal", "middle_intermediate"],
        fingertip_link_names=[
            "thumb_distal",
            "index_intermediate",
            "middle_intermediate",
            "ring_intermediate",
            "pinky_intermediate",
        ],
    )
    g1_inspire_right = RobotConfig(
        name="g1_inspire_right",
        urdf_or_mjcf_path=str(g1_inspire_assets.URDF_PATH),
        contact_points_path=str(g1_inspire_assets.CONTACT_POINTS_PATH),
        collision_spheres_path=str(g1_inspire_assets.COLLISION_SPHERE_PATH),
        self_collision_ignore_path=str(g1_inspire_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
        base_type="fixed",
        init_position=(-0.4, 0.0, 0.0),
        init_qpos=tuple([0.0] * 41),
        ee_coord_spec=HandCoordinateSpec(palm_forward_axis="-x", four_fingers_up_axis="-y"),
        thumb_link_names=["R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"],
        middle_link_names=["R_middle_proximal", "R_middle_intermediate"],
        fingertip_link_names=[
            "R_thumb_distal",
            "R_index_intermediate",
            "R_middle_intermediate",
            "R_ring_intermediate",
            "R_pinky_intermediate",
        ],
        hand_base_link_name="R_hand_base_link",
        ee_link_names=("R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"),
        arm_base_link_name="torso_link",
        workspace_origin=(-0.4, -0.4, 0.3),
    )
    return {
        "mano_right": mano_right,
        "xhand_right": xhand_right,
        "xarm7_xhand_right": xarm7_xhand_right,
        "inspire_hand_right": inspire_hand_right,
        "g1_inspire_right": g1_inspire_right,
    }


def make_optim_configs() -> Dict[str, BenchGraspOptimConfig]:
    xhand_mu = (1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3)
    inspire_mu = (0.3, 0.05, 0.6, 0.6, 0.6, 0.6)
    return {
        "mano_right": BenchGraspOptimConfig(optim_iters=6000),
        "xhand_right": BenchGraspOptimConfig(
            optim_iters=10000, optim_spen_threshold=0.025, init_distance_upper=0.15, init_hand_qpos_mu=xhand_mu
        ),
        "xarm7_xhand_right": BenchGraspOptimConfig(
            optim_iters=10000, optim_spen_threshold=0.025, init_distance_upper=0.15, init_hand_qpos_mu=xhand_mu
        ),
        "inspire_hand_right": BenchGraspOptimConfig(optim_iters=10000, init_hand_qpos_mu=inspire_mu),
        "g1_inspire_right": BenchGraspOptimConfig(optim_iters=10000, init_hand_qpos_mu=inspire_mu),
    }


# ---------------------------------------------------------------------------
# Robot loader (Phase 2)
# ---------------------------------------------------------------------------


def setup_robot(robot_config: RobotConfig, device_str: str) -> Tuple[GraspRobot, wp.array, wp.array]:
    """Build a robot + contact candidates via GraspRobot (from_config; get_contact_candidates: JSON or mesh-sampled).

    Returns:
        robot, contact_candidates_wp, contact_candidate_link_indices_wp
    """
    robot = GraspRobot.from_config(robot_config, load_meshes=True, load_collision_spheres=True, device=device_str)
    contact_centers, contact_link_indices = (
        t.cpu().numpy() for t in GraspRobot.get_contact_candidates(robot.config, robot.device)
    )

    contact_candidates_wp = wp.from_numpy(contact_centers.astype(np.float32), dtype=wp.vec3, device=device_str)
    contact_link_indices_wp = wp.from_numpy(contact_link_indices.astype(np.int32), dtype=wp.int32, device=device_str)

    return robot, contact_candidates_wp, contact_link_indices_wp


# ---------------------------------------------------------------------------
# Scene-geom backends (Phase 3)
# ---------------------------------------------------------------------------


def _sdf_token(value: float) -> str:
    """Match graspkit `sdf_float_token` so cached `.nvdb` filenames line up (e.g. `0.25`->`025`, `0.005`->`005`)."""
    text = f"{value:g}"
    if text.startswith("0."):
        digits = text[2:]
        if value >= 0.01:
            return f"0{digits}"
        return digits
    return text.replace(".", "p")


def _scaled_trimesh(mesh_path: str, scale: float) -> trimesh.Trimesh:
    """Load a mesh and scale vertices in place. Trimesh-only — no caching."""
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    return mesh


def build_scene_pair(
    scene_descs: List[SceneDescription],
    backend: Literal["mesh", "sdf_volume"],
    device_str: str,
    sdf_padding: float = 0.25,
    sdf_resolution: int = 64,
    sdf_voxel_size: Optional[float] = None,
    ground_plane_z: Optional[float] = None,
) -> Tuple[WarpScene, WarpScene, WarpScene, WarpScene, float, float]:
    """Build (target_geom, scene_geom, init_target_geom, init_scene_geom, bake_time, sdf_file_size_mb).

    - target_geom / scene_geom: backend-dependent (mesh or SDF), used by the optimizer.
    - init_*_geom: always mesh-backed, matching graspkit's SceneGeom (SDF here would drift the init).
    """
    n_scenes = len(scene_descs)
    target_world_poses, target_scales, target_mesh_paths = [], [], []
    # Per-scene box and plane groups (typed geoms carry one shape kind each).
    box_half_extents, box_poses, box_offsets = [], [], [0]
    plane_poses, plane_offsets = [], [0]
    for d in scene_descs:
        target_md = d.meshes[d.target_actor_id]
        target_mesh_paths.append(resolve_mesh_path(target_md.collision_mesh_path))
        target_world_poses.append(
            pose7_to_tf_mat(np.asarray([*target_md.position, *target_md.quaternion], dtype=np.float32))
        )
        target_scales.append(float(target_md.scales[0]))
        for b in d.boxes.values():
            box_half_extents.append(0.5 * np.asarray(b.size, dtype=np.float32))
            box_poses.append(pose7_to_tf_mat(np.asarray([*b.position, *b.quaternion], dtype=np.float32)))
        box_offsets.append(len(box_half_extents))
        if ground_plane_z is not None:
            gp_pose = np.eye(4, dtype=np.float32)
            gp_pose[2, 3] = float(ground_plane_z)
            plane_poses.append(gp_pose)
        plane_offsets.append(len(plane_poses))

    poses_wp = wp.from_numpy(np.stack(target_world_poses, axis=0), dtype=wp.mat44, device=device_str)
    scales_wp = wp.from_numpy(np.asarray(target_scales, dtype=np.float32), dtype=wp.float32, device=device_str)
    offsets = np.arange(n_scenes + 1, dtype=np.int32)

    def _add_target(scene: WarpScene, sdf_volumes: Optional[List[SdfVolume]]) -> None:
        if sdf_volumes is None:
            scene.add(MeshGeom(target_mesh_paths, offsets, poses=poses_wp, scales=scales_wp))
        else:
            scene.add(VolumeGeom(sdf_volumes, offsets, poses=poses_wp, scales=scales_wp))

    def _add_primitives(scene: WarpScene) -> None:
        if box_half_extents:
            box_poses_wp = wp.from_numpy(np.stack(box_poses, axis=0), dtype=wp.mat44, device=device_str)
            scene.add(BoxGeom(np.stack(box_half_extents), np.asarray(box_offsets, dtype=np.int32), poses=box_poses_wp))
        if plane_poses:
            plane_poses_wp = wp.from_numpy(np.stack(plane_poses, axis=0), dtype=wp.mat44, device=device_str)
            scene.add(PlaneGeom(np.asarray(plane_offsets, dtype=np.int32), poses=plane_poses_wp))

    t0 = time.perf_counter()
    sdf_volumes: Optional[List[SdfVolume]] = None
    sdf_file_size_mb = 0.0
    if backend == "sdf_volume":
        sdf_volumes = []
        total_grid_bytes = 0
        for path, scale in zip(target_mesh_paths, target_scales):
            mesh_path = Path(path)
            tag = (
                f"pad{_sdf_token(sdf_padding)}_res{sdf_resolution}"
                if sdf_voxel_size is None
                else f"pad{_sdf_token(sdf_padding)}_vox{_sdf_token(sdf_voxel_size)}"
            )
            nvdb_path = mesh_path.with_name(f"{mesh_path.stem}_{tag}.nvdb")
            mesh = _scaled_trimesh(path, scale)
            padded_min = mesh.bounds[0].astype(np.float32) - sdf_padding
            padded_max = mesh.bounds[1].astype(np.float32) + sdf_padding
            voxel = (
                sdf_voxel_size
                if sdf_voxel_size is not None
                else float((padded_max - padded_min).max()) / sdf_resolution
            )
            if nvdb_path.exists():
                with open(nvdb_path, "rb") as f:
                    volume = wp.Volume.load_from_nvdb(f, device=device_str)
                sv = SdfVolume(volume=volume, aabb_min=padded_min, aabb_max=padded_max, padding=float(sdf_padding))
            else:
                sv = SdfVolume.from_mesh(mesh, voxel_size=voxel, padding=sdf_padding, device=device_str)
            sdf_volumes.append(sv)
            ext = padded_max - padded_min
            ni, nj, nk = (int(np.ceil(ext[k] / voxel)) + 1 for k in range(3))
            total_grid_bytes += ni * nj * nk * 4
        sdf_file_size_mb = total_grid_bytes / (1024.0 * 1024.0) / n_scenes

    target_geom = WarpScene(num_scenes=n_scenes, device=device_str)
    _add_target(target_geom, sdf_volumes)

    scene_geom = WarpScene(num_scenes=n_scenes, device=device_str)
    _add_target(scene_geom, sdf_volumes)
    _add_primitives(scene_geom)

    bake_time = time.perf_counter() - t0

    if backend == "mesh":
        init_target_geom = target_geom
        init_scene_geom = scene_geom
    else:
        init_target_geom = WarpScene(num_scenes=n_scenes, device=device_str)
        _add_target(init_target_geom, None)
        init_scene_geom = WarpScene(num_scenes=n_scenes, device=device_str)
        _add_target(init_scene_geom, None)
        _add_primitives(init_scene_geom)

    return target_geom, scene_geom, init_target_geom, init_scene_geom, bake_time, sdf_file_size_mb


# ---------------------------------------------------------------------------

_PRESET_TO_GRASPKIT_EXP: Dict[str, str] = {
    "mano_right": "mano_right_dex1b_v0_4_mini_floating",
    "xhand_right": "xhand_right_dex1b_v0_4_mini_floating",
    "xarm7_xhand_right": "xarm7_xhand_right_dex1b_v0_4_mini_tabletop",
    "g1_inspire_right": "g1_inspire_right_dex1b_v0_4_mini_tabletop",
}


def _build_graspkit_scenes(scene_descriptions: List[SceneDescription]) -> List[SceneDescription]:
    """Resolve graspkit mesh paths to absolute robokit paths; same dataclass, only path strings differ."""
    g_scenes = []
    for s in scene_descriptions:
        g_meshes = {
            k: MeshDescription(
                actor_id=m.actor_id,
                collision_mesh_path=resolve_mesh_path(m.collision_mesh_path),
                scales=m.scales,
                position=m.position,
                quaternion=m.quaternion,
                visual_mesh_path=resolve_mesh_path(m.visual_mesh_path) if m.visual_mesh_path else None,
                bounds=m.bounds,
            )
            for k, m in s.meshes.items()
        }
        g_boxes = {
            k: BoxDescription(actor_id=b.actor_id, size=b.size, position=b.position, quaternion=b.quaternion)
            for k, b in s.boxes.items()
        }
        g_scenes.append(
            SceneDescription(scene_id=s.scene_id, boxes=g_boxes, meshes=g_meshes, target_actor_id=s.target_actor_id)
        )
    return g_scenes


def plan_with_graspkit(
    grasp_qpos: torch.Tensor,
    robot_cfg: RobotConfig,
    scene_descriptions: List[SceneDescription],
    device_str: str,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Bridges to graspkit's motion planner; returns (full_traj_qpos, pre_traj_penetration, mp_time, mp_sr)."""
    from graspkit.configs.experiment import build_experiment_config
    from graspkit.datasets.graspkit_data import GraspKitData
    from graspkit.stages.motion_plan import build_motion_planner

    exp_cfg = build_experiment_config(_PRESET_TO_GRASPKIT_EXP[robot_cfg.name])
    mp_cfg = exp_cfg.traj_builder.motion_planner
    if robot_cfg.base_type == "floating":
        eval_robot_cfg = exp_cfg.robot_config or exp_cfg.traj_builder.fixed_base_robot_config
    else:
        eval_robot_cfg = exp_cfg.traj_builder.fixed_base_robot_config

    g_scenes = _build_graspkit_scenes(scene_descriptions)
    batch = GraspKitData(
        scene_ids=[s.scene_id for s in g_scenes],
        scene_descriptions=g_scenes,
        grasp_qpos=grasp_qpos,
    )

    device = grasp_qpos.device
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    motion_planner = build_motion_planner(mp_cfg, robot_config=eval_robot_cfg, device=device)
    out_batch = motion_planner.plan(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mp_time = time.perf_counter() - t0
    full_traj = out_batch.full_traj_qpos
    pre_pen = out_batch.pre_traj_penetration
    mp_sr = (pre_pen < 0.001).float().mean().item() * 100.0
    return full_traj, pre_pen, mp_time, mp_sr


def run_sim(
    full_traj_qpos: torch.Tensor,
    scene_descriptions: List[SceneDescription],
    robot_cfg: RobotConfig,
    num_envs: int,
) -> Dict[str, float]:
    """Hybrid sim eval: bridges to graspkit's Evaluator via SceneDescription translation; returns {sim_time, sim_sr}."""
    from dataclasses import replace as dc_replace

    from graspkit.configs.experiment import build_experiment_config
    from graspkit.datasets.graspkit_data import GraspKitData
    from graspkit.envs import MsEnv

    g_scenes = _build_graspkit_scenes(scene_descriptions)

    exp_cfg = build_experiment_config(_PRESET_TO_GRASPKIT_EXP[robot_cfg.name])
    # grasp_optim.evaluator = in-loop stress filter; reference numbers were measured with it, so keep it.
    eval_cfg = exp_cfg.grasp_optim.evaluator
    if robot_cfg.base_type == "floating":
        eval_robot_cfg = exp_cfg.robot_config or exp_cfg.traj_builder.fixed_base_robot_config
    else:
        eval_robot_cfg = exp_cfg.traj_builder.fixed_base_robot_config
    assert eval_cfg is not None, f"evaluator config is None for {robot_cfg.name}"
    # Match graspkit bench: only override num_envs (preserve preset num_seeds, perturbation, etc.).
    eval_cfg = dc_replace(eval_cfg, num_envs=num_envs)

    batch = GraspKitData(
        scene_ids=[s.scene_id for s in g_scenes],
        scene_descriptions=g_scenes,
        full_traj_qpos=full_traj_qpos,
    )
    device = full_traj_qpos.device
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    evaluator = MsEnv(eval_cfg, robot_config=eval_robot_cfg, device=device)
    infos = evaluator.evaluate(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = time.perf_counter() - t_start
    evaluator.close()

    success = infos["success"] if isinstance(infos, dict) else infos[0]["success"]
    sim_sr = success.float().mean().item() * 100.0
    return {"sim_time": sim_time, "sim_sr": sim_sr}
