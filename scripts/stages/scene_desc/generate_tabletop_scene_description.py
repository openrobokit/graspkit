"""
CUDA_VISIBLE_DEVICES=2 uv run scripts/stages/scene_desc/generate_tabletop_scene_description.py dex1b_v0.3_small_world_tabletop_mini_train
"""

import graspkit.mani_skill  # noqa: F401 I001

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Literal

import gymnasium as gym
import numpy as np
import sapien.physx
import torch
from robokit.xform.numpy import axis_angle_to_quaternion, quaternion_raw_multiply
import tyro
from rich.progress import track

from graspkit.assets.objects import dexgraspnet_official_splits
from graspkit.configs.experiment_config import EnvConfig
from graspkit.assets.objects.discovery import find_dexdiffuser_assets_v2, find_dexgraspnet_assets
from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.tensor_utils import batchify_tensor, padding_tensor

logger = logging.getLogger("graspkit")


@dataclass
class Config:
    source_type: Literal["meshes", "canonical"] = "meshes"
    """Source for generation: 'meshes' enumerates meshes directly; 'canonical' loads canonical tabletop scenes."""
    canonicalized_pose: bool = False
    """Whether to use canonicalized poses (no randomization, num_candidates_per_canonical_scene = len(orientations))."""

    # mesh roster (used when source_type='meshes')
    mesh_sources: list[Literal["dexgraspnet", "dexdiffuser_v2"]] = field(default_factory=lambda: ["dexgraspnet"])
    """Object mesh sources to enumerate for canonical generation."""
    scales: list[float] = field(default_factory=lambda: [0.06, 0.08, 0.1, 0.12, 0.15])
    """Scales applied per dexgraspnet object (dexdiffuser_v2 always uses scale 1.0)."""
    dexgraspnet_split_dir: str = str(dexgraspnet_official_splits.DIR)
    """Path to the dexgraspnet split directory."""
    dexdiffuser_sources: list[str] | None = None
    """Subset of dexdiffuser_v2 source dirs to scan (None = all: egad, kit, contactdb, ycb)."""
    max_num_object_instances: int | None = None
    """Optional cap on the number of (object, scale) instances before orientation expansion (None = all)."""
    max_num_objects: int | None = None
    """Optional cap on distinct objects per source, applied before the x-scale expansion (None = all)."""
    canonical_scene_description_paths: list[str] = field(default_factory=list)
    """Paths to canonical tabletop scene description files (used when source_type='canonical')."""
    output_path: str = str(DATA_ROOT / "scene_description/dex1b_v0.3/tabletop.msgpack")
    """Path to save the scene description."""
    existing_scene_description_path: str | None = None
    """Optional existing scene description msgpack to merge new scenes into before saving to `output_path`."""
    split: Literal["train", "test"] = "train"
    """Split to use from the source scene description file."""

    world_size: int = 1
    """Total number of parallel processes."""
    rank: int = 0
    """Rank of the current process (0-indexed)."""

    num_envs: int = 4096
    """Number of environments to generate."""
    num_steps: int = 500
    """Number of simulation steps."""
    num_candidates_per_canonical_scene: int = 120
    """Number of randomized copies generated per canonical snapshot (NOT per object)."""
    max_num_scenes: int = 10_000_000
    """Maximum number of scenes to generate."""

    # valid object bounds
    valid_xy_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.15, 0.15),
        (-0.15, 0.15),
    )
    """Bounds of valid object translation in the xy plane, as ((x_min, x_max), (y_min, y_max))."""
    min_object_height_threshold: float = 0.02
    """Minimum height threshold for valid objects (z-axis upper bound)."""
    max_object_volume_threshold: float = 0.003
    """Maximum volume threshold for valid objects."""
    max_object_xy_area_threshold: float = 0.025
    """Maximum xy area threshold for valid objects."""

    # randomization parameters
    enable_xy_translation_randomization: bool = True
    """Whether to enable translation randomization in the xy plane."""
    xy_translation_randomization_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.075, 0.075),
        (-0.075, 0.075),
    )
    """Bounds of the translation randomization in the xy plane, as ((x_min, x_max), (y_min, y_max))."""
    orientations: list[Literal["none", "x90", "y90"]] = field(default_factory=lambda: ["none", "x90", "y90"])
    """Orientations to apply to the objects."""
    enable_z_rotation_randomization: bool = True
    """Whether to enable rotation randomization in the z axis."""
    z_rotation_randomization_bounds: tuple[float, float] = (-np.pi, np.pi)
    """Bounds of the rotation randomization in the z axis."""

    # rebalancing
    enable_height_rebalancing: bool = True
    """Whether to enable height rebalancing."""
    height_rebalancing_bounds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.02, 0.05),
            (0.05, 0.09),
            (0.09, 0.15),
            (0.15, 0.25),
        ]
    )
    """Bounds of the height rebalancing."""
    enable_object_rebalancing: bool = True
    """Whether to enable object rebalancing."""
    max_scene_count_per_object: int = 10
    """Maximum number of scenes per object."""

    # per-scene absolute table (x, y, top-z) sampling (YCB-style world randomization)
    enable_table_pose_randomization: bool = False
    """If True, sample a per-scene table pose from table_x_range / table_y_range / table_height_range."""
    table_x_range: tuple[float, float] = (0.0, 0.0)
    """Uniform range (meters) for the sampled table center x in world frame."""
    table_y_range: tuple[float, float] = (0.0, 0.0)
    """Uniform range (meters) for the sampled table center y in world frame."""
    table_height_range: tuple[float, float] = (0.3, 0.75)
    """Uniform range (meters) for the sampled table TOP z in world frame."""

    table_size: tuple[float, float, float] = (2.418, 1.209, 0.9196429)
    """Size of the table, aligned with ManiSkill's table builder (scene_builder/table/scene_builder.py#L33)."""
    table_pos: tuple[float, float, float] = (-0.12, 0.0, -0.9196429 / 2)
    """Position of the table."""
    table_quat: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
    """Quaternion of the table."""

    def __post_init__(self):
        if self.canonicalized_pose:
            self.enable_xy_translation_randomization = False
            self.enable_z_rotation_randomization = False
            self.num_candidates_per_canonical_scene = len(self.orientations)

        # Validate source_type configuration
        if self.source_type == "meshes" and not self.mesh_sources:
            raise ValueError("mesh_sources must be provided when source_type='meshes'")
        if self.source_type == "canonical" and not self.canonical_scene_description_paths:
            raise ValueError("canonical_scene_description_paths must be provided when source_type='canonical'")


ORIENTATION_MAPPING = {
    "none": (1.0, 0.0, 0.0, 0.0),
    "x90": tuple(axis_angle_to_quaternion(np.array([np.pi / 2.0, 0.0, 0.0])).tolist()),
    "y90": tuple(axis_angle_to_quaternion(np.array([0.0, np.pi / 2.0, 0.0])).tolist()),
}


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def filter_invalid_scene_descriptions(
    scene_desc_candidates: list[SceneDescription], cfg: Config, height_offset: float = 0.0
) -> list[SceneDescription]:
    """Filter scene descriptions failing validity criteria; `valid_xy_bounds` is table-relative when randomized."""
    logger.info(f"Filtering scene descriptions from {len(scene_desc_candidates)} candidates...")
    valid_scene_desc_candidates = []
    for scene_desc in track(scene_desc_candidates, description="Filtering...", disable=True):
        bounds = scene_desc.meshes["object"].get_bounds()
        volume = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0]) * (bounds[2][1] - bounds[2][0])
        xy_area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])

        if cfg.enable_table_pose_randomization and "table" in scene_desc.boxes:
            table = scene_desc.boxes["table"]
            tx, ty = table.position[0], table.position[1]
            tz = table.position[2] + table.size[2] / 2  # table top z; height threshold is table-relative
        else:
            tx, ty, tz = 0.0, 0.0, 0.0

        valid_xy = (
            bounds[0][0] >= cfg.valid_xy_bounds[0][0] + tx
            and bounds[0][1] <= cfg.valid_xy_bounds[0][1] + tx
            and bounds[1][0] >= cfg.valid_xy_bounds[1][0] + ty
            and bounds[1][1] <= cfg.valid_xy_bounds[1][1] + ty
        )
        valid_z = bounds[2][1] >= cfg.min_object_height_threshold + height_offset + tz
        valid_volume = volume <= cfg.max_object_volume_threshold
        valid_xy_area = xy_area <= cfg.max_object_xy_area_threshold

        if valid_xy and valid_z and valid_volume and valid_xy_area:
            valid_scene_desc_candidates.append(scene_desc)

    logger.info(
        f"Removed {len(scene_desc_candidates) - len(valid_scene_desc_candidates)} invalid scene descriptions..."
    )
    if len(valid_scene_desc_candidates) == 0:
        raise ValueError("No valid scene descriptions found")
    return valid_scene_desc_candidates


def rebalance_scene_descriptions_based_on_heights(
    scene_desc_candidates: list[SceneDescription], cfg: Config
) -> list[SceneDescription]:
    """Rebalance scene descriptions based on z range with optimized grouping."""
    logger.info(f"Rebalancing {len(scene_desc_candidates)} scene descriptions based on heights...")
    group_bounds = np.array(cfg.height_rebalancing_bounds)
    group_centers = group_bounds.mean(axis=1)
    groups: dict[tuple[float, float], list[SceneDescription]] = {
        tuple(bounds): [] for bounds in cfg.height_rebalancing_bounds
    }

    for scene in scene_desc_candidates:
        bounds = scene.meshes["object"].get_bounds()
        obj_height = bounds[2][1] - bounds[2][0]  # object height; table-z jitter must not shift the bin
        condition = (group_bounds[:, 0] <= obj_height) & (obj_height < group_bounds[:, 1])
        valid_idx = np.where(condition)[0]
        if valid_idx.size > 0:
            group_key = tuple(cfg.height_rebalancing_bounds[int(valid_idx[0])])
        else:
            diffs = np.abs(obj_height - group_centers)
            idx = int(np.argmin(diffs))
            group_key = tuple(cfg.height_rebalancing_bounds[idx])
        groups[group_key].append(scene)

    if any(len(v) == 0 for v in groups.values()):
        raise RuntimeError("Empty groups found after rebalancing")

    min_size = min(len(scenes) for scenes in groups.values())
    rng = np.random.RandomState(42)
    balanced_scenes: list[SceneDescription] = []

    for _, scenes in groups.items():
        coll_mesh_dict: dict[str, list[SceneDescription]] = defaultdict(list)
        for s in scenes:
            key = s.meshes["object"].collision_mesh_path
            coll_mesh_dict[key].append(s)

        subsampled: list[SceneDescription] = []
        object_keys = list(coll_mesh_dict.keys())
        rng.shuffle(object_keys)

        while len(subsampled) < min_size:
            round_selected = 0
            for key in object_keys:
                if coll_mesh_dict[key]:
                    scene_choice = rng.choice(coll_mesh_dict[key])
                    coll_mesh_dict[key].remove(scene_choice)
                    subsampled.append(scene_choice)
                    round_selected += 1
                    if len(subsampled) == min_size:
                        break
            if round_selected == 0:
                break
        if len(subsampled) != min_size:
            raise ValueError(f"subsampled {len(subsampled)} scene descriptions, expected {min_size}")
        balanced_scenes.extend(subsampled)

    logger.info(f"Rebalanced to {len(balanced_scenes)} scene descriptions...")
    return balanced_scenes


def rebalance_scene_descriptions_based_on_object_counts(
    scene_desc_candidates: list[SceneDescription],
    cfg: Config,
    mode: Literal["mesh_path", "floating_scene_id", "canonical_scene_id"] = "mesh_path",
) -> list[SceneDescription]:
    """Rebalance scene descriptions based on object counts."""
    logger.info(f"Rebalancing {len(scene_desc_candidates)} scene descriptions based on {mode} counts...")
    scene_groups: dict[str, list[SceneDescription]] = defaultdict(list)
    for scene in scene_desc_candidates:
        if mode == "mesh_path":
            key = scene.meshes["object"].collision_mesh_path
        elif mode == "floating_scene_id":
            key = scene.floating_scene_id
        elif mode == "canonical_scene_id":
            key = scene.canonical_tabletop_scene_id
        else:
            raise ValueError(f"Invalid rebalance mode: {mode}")

        if key is None:
            key = "None"
        scene_groups[key].append(scene)

    rebalanced_scenes: list[SceneDescription] = []
    rng = np.random.RandomState(42)
    for key, scenes in scene_groups.items():
        if len(scenes) > cfg.max_scene_count_per_object:
            selected_indices = rng.choice(len(scenes), size=cfg.max_scene_count_per_object, replace=False)
            for idx in selected_indices:
                rebalanced_scenes.append(scenes[idx])
        else:
            rebalanced_scenes.extend(scenes)
    logger.info(f"Rebalanced to {len(rebalanced_scenes)} scene descriptions.")
    return rebalanced_scenes


def save_valid_scene_descriptions(valid_scene_descriptions, cfg: Config):
    """Save scenes to cfg.output_path, optionally merging into an existing scene description file."""
    scene_dict = {scene_desc.scene_id: scene_desc for scene_desc in valid_scene_descriptions}
    if cfg.split == "train":
        scenes_to_save = {"train": scene_dict, "test": {}}
    else:
        scenes_to_save = {"train": {}, "test": scene_dict}

    if cfg.existing_scene_description_path:
        existing_splits = SceneDescription.load_scene_descriptions_msgpack(
            cfg.existing_scene_description_path, return_splits=True
        )
        logger.info(
            f"Merging into existing scene description '{cfg.existing_scene_description_path}' "
            f"(train={len(existing_splits['train'])}, test={len(existing_splits['test'])})"
        )
        # new scenes win on scene_id collisions
        scenes_to_save["train"] = {**existing_splits["train"], **scenes_to_save["train"]}
        scenes_to_save["test"] = {**existing_splits["test"], **scenes_to_save["test"]}

    output_path = cfg.output_path
    if cfg.world_size > 1:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_w{cfg.world_size}_r{cfg.rank}{ext}"

    logger.info(
        f"Saving {len(scenes_to_save['train'])} train + {len(scenes_to_save['test'])} test "
        f"scene descriptions to {output_path}"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SceneDescription.save_scene_descriptions_msgpack(scenes_to_save, msgpack_path=output_path)


def generate_canonical_from_meshes(cfg: Config):
    """Enumerate object meshes directly, convert to table-top, simulate dropping, validate rest poses, and save."""
    # Build the object roster directly from mesh sources (no floating msgpack).
    rng = np.random.RandomState(42)
    instances: list[tuple[str, MeshDescription]] = []  # (scene_id_base, object mesh at origin)
    for source in cfg.mesh_sources:
        if source == "dexgraspnet":
            from graspkit.assets.objects import dexgraspnet

            names, coll_paths = find_dexgraspnet_assets(str(dexgraspnet.DATA_DIR), cfg.dexgraspnet_split_dir, cfg.split)
            visual_paths, scales = [None] * len(names), cfg.scales
        else:  # dexdiffuser_v2
            from graspkit.assets.objects import dex_diffuser_v2

            names, coll_paths, visual_paths = find_dexdiffuser_assets_v2(
                str(dex_diffuser_v2.DATA_DIR), cfg.dexdiffuser_sources
            )
            scales = [1.0]
        logger.info(f"Found {len(names)} {source} objects for '{cfg.split}' split")
        if cfg.max_num_objects is not None and len(names) > cfg.max_num_objects:
            sel = sorted(rng.choice(len(names), cfg.max_num_objects, replace=False))
            names = [names[i] for i in sel]
            coll_paths = [coll_paths[i] for i in sel]
            visual_paths = [visual_paths[i] for i in sel]
            logger.info(f"Subsampled {source} to {len(names)} distinct objects (max_num_objects)")
        for name, coll_path, visual_path in zip(names, coll_paths, visual_paths):
            for inst_idx, scale in enumerate(scales):
                obj_mesh_desc = MeshDescription(
                    actor_id="object",
                    collision_mesh_path=os.path.relpath(coll_path, PROJECT_ROOT),
                    visual_mesh_path=os.path.relpath(visual_path, PROJECT_ROOT) if visual_path is not None else None,
                    scales=(scale, scale, scale),
                )
                instances.append((f"{source}_{name}_inst_{inst_idx}_scale_{scale:.3f}", obj_mesh_desc))

    if cfg.max_num_object_instances is not None and len(instances) > cfg.max_num_object_instances:
        instances = [
            instances[i] for i in sorted(rng.choice(len(instances), cfg.max_num_object_instances, replace=False))
        ]
        logger.info(f"Subsampled roster to {len(instances)} object instances")

    if cfg.world_size > 1:
        instances = sorted(instances, key=lambda x: x[0])
        instances = [inst for i, inst in enumerate(instances) if i % cfg.world_size == cfg.rank]
        logger.info(f"Rank {cfg.rank}/{cfg.world_size}: processing {len(instances)} object instances")

    table_desc = BoxDescription(
        actor_id="table", size=cfg.table_size, position=cfg.table_pos, quaternion=cfg.table_quat
    )

    scene_desc_candidates: list[SceneDescription] = []
    for scene_id_base, ori_obj_mesh_desc in track(instances, description="Converting meshes to canonical"):
        for orientation in cfg.orientations:
            ori_quat = ORIENTATION_MAPPING[orientation]
            position = (0.0, 0.0, 0.0)

            initial_obj_mesh_desc = replace(
                ori_obj_mesh_desc, actor_id="object", position=position, quaternion=ori_quat, bounds=None
            )

            bounds = initial_obj_mesh_desc.get_bounds()
            z_offset = -bounds[2][0] + 0.005
            updated_position = (position[0], position[1], z_offset)
            updated_bounds = (bounds[0], bounds[1], (bounds[2][0] + z_offset, bounds[2][1] + z_offset))

            new_obj_mesh_desc = replace(initial_obj_mesh_desc, position=updated_position, bounds=updated_bounds)
            new_scene_id = f"{scene_id_base}_ori_{orientation}_canonical_tabletop"
            new_scene_desc = SceneDescription(
                floating_scene_id=scene_id_base,
                scene_id=new_scene_id,
                boxes={table_desc.actor_id: table_desc},
                meshes={new_obj_mesh_desc.actor_id: new_obj_mesh_desc},
                target_actor_id="object",
            )
            scene_desc_candidates.append(new_scene_desc)

    logger.info(
        f"Created {len(scene_desc_candidates)} canonical table-top scene candidates "
        f"from {len(instances)} object instances"
    )

    scene_desc_candidates = filter_invalid_scene_descriptions(scene_desc_candidates, cfg, height_offset=0.005)
    if cfg.enable_height_rebalancing:
        scene_desc_candidates = rebalance_scene_descriptions_based_on_heights(scene_desc_candidates, cfg)

    env = gym.make(
        "DexGrasp",
        config=EnvConfig(load_scenes=True, object_mass=1.0),
        robot_uids="shadow_right_mj_floating_base",
        num_envs=cfg.num_envs,
        sim_backend="gpu",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
    )

    logger.info("Simulating...")
    rest_poses = []
    table_aabb_checked = False
    for batch_scene_desc in track(
        batchify_tensor(scene_desc_candidates, cfg.num_envs),
        description="Simulating...",
        total=math.ceil(len(scene_desc_candidates) / cfg.num_envs),
    ):
        batch_scene_desc, batch_mask = padding_tensor(batch_scene_desc, cfg.num_envs)
        env.reset(options={"scene_descriptions": batch_scene_desc, "reconfigure": True}, seed=0)
        if not table_aabb_checked:
            table_physx = env.unwrapped.table_scene.table._objs[0].find_component_by_type(
                sapien.physx.PhysxRigidBodyComponent
            )
            table_aabb = table_physx.compute_global_aabb_tight()
            assert np.isclose(table_aabb[1, 2], 0.0, atol=1e-3), f"Table top z must be 0.0, got {table_aabb[1, 2]}"
            table_aabb_size = table_aabb[1] - table_aabb[0]
            assert np.allclose(table_aabb_size, cfg.table_size, atol=1e-2), (
                f"Table AABB size {table_aabb_size} doesn't match config size {cfg.table_size}"
            )
            table_aabb_checked = True
        cur_robot_pose = env.unwrapped.agent.robot.get_pose().raw_pose
        cur_robot_pose[..., 2] = 10.0
        env.unwrapped.agent.robot.set_pose(cur_robot_pose)
        env.unwrapped.scene._gpu_apply_all()
        for _ in range(cfg.num_steps):
            obs, _, _, _, _ = env.step(None)
        batch_rest_poses = obs["extra"]["actor_pose"][batch_mask]
        rest_poses.append(batch_rest_poses)
    rest_poses = torch.cat(rest_poses, dim=0)
    rest_poses[..., 3:] = standardize_quaternion(rest_poses[..., 3:])

    for scene_desc, rest_pose in zip(scene_desc_candidates, rest_poses):
        updated_position = tuple(rest_pose[..., :3].tolist())
        updated_quaternion = tuple(rest_pose[..., 3:].tolist())
        old_mesh = scene_desc.meshes["object"]
        updated_mesh = replace(old_mesh, position=updated_position, quaternion=updated_quaternion, bounds=None)
        scene_desc.meshes["object"] = replace(updated_mesh, bounds=updated_mesh.get_bounds())

    logger.info("Checking...")
    rest_poses_check = []
    for batch_scene_desc in track(
        batchify_tensor(scene_desc_candidates, cfg.num_envs),
        description="Checking...",
        total=math.ceil(len(scene_desc_candidates) / cfg.num_envs),
    ):
        batch_scene_desc, batch_mask = padding_tensor(batch_scene_desc, cfg.num_envs)
        env.reset(options={"scene_descriptions": batch_scene_desc, "reconfigure": True}, seed=0)
        cur_robot_pose = env.unwrapped.agent.robot.get_pose().raw_pose
        cur_robot_pose[..., 2] = 10.0
        env.unwrapped.agent.robot.set_pose(cur_robot_pose)
        env.unwrapped.scene._gpu_apply_all()
        for _ in range(50):
            obs, _, _, _, _ = env.step(None)
        batch_rest_poses = obs["extra"]["actor_pose"][batch_mask]
        rest_poses_check.append(batch_rest_poses)
    rest_poses_check = torch.cat(rest_poses_check, dim=0)
    rest_poses_check[..., 3:] = standardize_quaternion(rest_poses_check[..., 3:])

    valid_pos = torch.norm(rest_poses[..., :3] - rest_poses_check[..., :3], dim=-1) < 5e-3
    valid_quat = torch.norm(rest_poses[..., 3:] - rest_poses_check[..., 3:], dim=-1) < 5e-2
    valid_obj = torch.logical_and(valid_pos, valid_quat).cpu()
    logger.info(f"Found {torch.logical_not(valid_obj).sum()} invalid objects out of {len(valid_obj)}")
    valid_scene_descriptions = [scene_desc for scene_desc, valid in zip(scene_desc_candidates, valid_obj) if valid]
    valid_scene_descriptions = filter_invalid_scene_descriptions(valid_scene_descriptions, cfg)
    if cfg.enable_object_rebalancing:
        valid_scene_descriptions = rebalance_scene_descriptions_based_on_object_counts(
            valid_scene_descriptions, cfg, mode="floating_scene_id"
        )
    if cfg.enable_height_rebalancing:
        valid_scene_descriptions = rebalance_scene_descriptions_based_on_heights(valid_scene_descriptions, cfg)
    rng = np.random.RandomState(42)
    max_num_scenes_for_rank = cfg.max_num_scenes // cfg.world_size if cfg.world_size > 1 else cfg.max_num_scenes
    if len(valid_scene_descriptions) > max_num_scenes_for_rank:
        logger.info(f"Subsampling {len(valid_scene_descriptions)} to {max_num_scenes_for_rank}...")
        valid_scene_descriptions = rng.choice(valid_scene_descriptions, max_num_scenes_for_rank, replace=False)
    inspect_scene_descriptions(valid_scene_descriptions, mode="floating_scene_id")

    save_valid_scene_descriptions(valid_scene_descriptions, cfg)


def generate_noncanonical_from_canonical(cfg: Config):
    """Load canonical tabletop scenes, apply xy translation and z rotation randomization, validate poses, and save."""
    # Load canonical tabletop scene descriptions from multiple paths
    all_canonical_scene_descriptions = {}
    for canonical_path in cfg.canonical_scene_description_paths:
        canonical_scene_descriptions_splits = SceneDescription.load_scene_descriptions_msgpack(
            canonical_path, return_splits=True
        )
        canonical_scene_descriptions = canonical_scene_descriptions_splits[cfg.split]
        all_canonical_scene_descriptions.update(canonical_scene_descriptions)
        logger.info(
            f"Loaded {len(canonical_scene_descriptions)} canonical scene descriptions from '{canonical_path}' (split: {cfg.split})"
        )

    logger.info(f"Total loaded: {len(all_canonical_scene_descriptions)} canonical scene descriptions")

    # Split scenes based on rank and world_size
    if cfg.world_size > 1:
        all_scene_ids = sorted(all_canonical_scene_descriptions.keys())
        scene_ids_for_rank = [scene_id for i, scene_id in enumerate(all_scene_ids) if i % cfg.world_size == cfg.rank]
        all_canonical_scene_descriptions = {
            scene_id: all_canonical_scene_descriptions[scene_id] for scene_id in scene_ids_for_rank
        }
        logger.info(f"Rank {cfg.rank}/{cfg.world_size}: processing {len(all_canonical_scene_descriptions)} scenes")

    # Create table description
    table_desc = BoxDescription(
        actor_id="table", size=cfg.table_size, position=cfg.table_pos, quaternion=cfg.table_quat
    )

    scene_desc_candidates: list[SceneDescription] = []
    rng = np.random.RandomState(42)
    for scene_id, canonical_scene_desc in track(
        all_canonical_scene_descriptions.items(), description="Generating non-canonical scenes"
    ):
        ori_obj_mesh_desc = canonical_scene_desc.meshes["object"]
        ori_position = ori_obj_mesh_desc.position
        ori_quaternion = ori_obj_mesh_desc.quaternion

        for scene_candidate_idx in range(cfg.num_candidates_per_canonical_scene):
            # per-scene absolute table pose (YCB-style); object re-centered at ori_position[2] above the new table top
            if cfg.enable_table_pose_randomization:
                table_rand_x = rng.uniform(*cfg.table_x_range)
                table_rand_y = rng.uniform(*cfg.table_y_range)
                table_rand_top_z = rng.uniform(*cfg.table_height_range)
                obj_base_x, obj_base_y, obj_base_z = table_rand_x, table_rand_y, table_rand_top_z + ori_position[2]
                scene_table_desc = BoxDescription(
                    actor_id="table",
                    size=cfg.table_size,
                    position=(table_rand_x, table_rand_y, table_rand_top_z - cfg.table_size[2] / 2),
                    quaternion=cfg.table_quat,
                )
            else:
                obj_base_x, obj_base_y, obj_base_z = ori_position
                scene_table_desc = table_desc

            # xy jitter on top of the base object position (intended to be within table footprint)
            if cfg.enable_xy_translation_randomization:
                rand_tl_bounds = cfg.xy_translation_randomization_bounds
                rand_x = rng.uniform(rand_tl_bounds[0][0], rand_tl_bounds[0][1])
                rand_y = rng.uniform(rand_tl_bounds[1][0], rand_tl_bounds[1][1])
            else:
                rand_x = rand_y = 0.0
            new_position = (obj_base_x + rand_x, obj_base_y + rand_y, obj_base_z)

            # Apply z rotation randomization
            if cfg.enable_z_rotation_randomization:
                rand_rot_bounds = cfg.z_rotation_randomization_bounds
                rand_z = rng.uniform(rand_rot_bounds[0], rand_rot_bounds[1])
                rand_z_quat = axis_angle_to_quaternion(np.array([0.0, 0.0, rand_z]))
                new_quaternion = tuple(quaternion_raw_multiply(rand_z_quat, np.array(ori_quaternion)).tolist())
            else:
                new_quaternion = ori_quaternion

            # Create new mesh description with randomized pose
            new_obj_mesh_desc = replace(
                ori_obj_mesh_desc, position=new_position, quaternion=new_quaternion, bounds=None
            )

            # Compute bounds for the new pose
            new_obj_mesh_desc = replace(new_obj_mesh_desc, bounds=new_obj_mesh_desc.get_bounds())

            # Create new scene description
            new_scene_id = scene_id.replace("canonical_tabletop", f"pose_{scene_candidate_idx}_tabletop")
            new_scene_desc = SceneDescription(
                floating_scene_id=canonical_scene_desc.floating_scene_id,
                canonical_tabletop_scene_id=scene_id,
                scene_id=new_scene_id,
                boxes={scene_table_desc.actor_id: scene_table_desc},
                meshes={new_obj_mesh_desc.actor_id: new_obj_mesh_desc},
                target_actor_id=canonical_scene_desc.target_actor_id,
            )
            scene_desc_candidates.append(new_scene_desc)

    logger.info(
        f"Created {len(scene_desc_candidates)} tabletop scene candidates "
        f"from {len(all_canonical_scene_descriptions)} canonical scenes"
    )

    # Pre-simulation filtering and rebalancing
    scene_desc_candidates = filter_invalid_scene_descriptions(scene_desc_candidates, cfg, height_offset=0.0)
    if cfg.enable_height_rebalancing:
        scene_desc_candidates = rebalance_scene_descriptions_based_on_heights(scene_desc_candidates, cfg)
    if cfg.enable_object_rebalancing:  # do not rebalance scene ids before height rebalancing
        scene_desc_candidates = rebalance_scene_descriptions_based_on_object_counts(
            scene_desc_candidates, cfg, mode="canonical_scene_id"
        )

    # Simulate to validate that objects keep their poses
    env = gym.make(
        "DexGrasp",
        config=EnvConfig(load_scenes=True, object_mass=1.0),
        robot_uids="shadow_right_mj_floating_base",
        num_envs=cfg.num_envs,
        sim_backend="gpu",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
    )

    logger.info("Simulating to check pose stability...")
    # Extract poses from scene descriptions
    scene_desc_poses = []
    for scene_desc in scene_desc_candidates:
        obj_mesh = scene_desc.meshes["object"]
        pose = list(obj_mesh.position) + list(obj_mesh.quaternion)
        scene_desc_poses.append(pose)
    scene_desc_poses = torch.tensor(scene_desc_poses, dtype=torch.float32)
    scene_desc_poses[..., 3:] = standardize_quaternion(scene_desc_poses[..., 3:])

    rest_poses = []
    table_aabb_checked = False
    for batch_scene_desc in track(
        batchify_tensor(scene_desc_candidates, cfg.num_envs),
        description="Simulating...",
        total=math.ceil(len(scene_desc_candidates) / cfg.num_envs),
    ):
        batch_scene_desc, batch_mask = padding_tensor(batch_scene_desc, cfg.num_envs)
        env.reset(options={"scene_descriptions": batch_scene_desc, "reconfigure": True}, seed=0)
        if not table_aabb_checked:
            table_physx = env.unwrapped.table_scene.table._objs[0].find_component_by_type(
                sapien.physx.PhysxRigidBodyComponent
            )
            table_aabb = table_physx.compute_global_aabb_tight()
            if not cfg.enable_table_pose_randomization:
                assert np.isclose(table_aabb[1, 2], 0.0, atol=1e-3), f"Table top z must be 0.0, got {table_aabb[1, 2]}"
            table_aabb_size = table_aabb[1] - table_aabb[0]
            assert np.allclose(table_aabb_size, cfg.table_size, atol=1e-2), (
                f"Table AABB size {table_aabb_size} doesn't match config size {cfg.table_size}"
            )
            table_aabb_checked = True

        # Move robot away
        cur_robot_pose = env.unwrapped.agent.robot.get_pose().raw_pose
        cur_robot_pose[..., 2] = 10.0
        env.unwrapped.agent.robot.set_pose(cur_robot_pose)
        env.unwrapped.scene._gpu_apply_all()

        # Run simulation
        for _ in range(cfg.num_steps):
            obs, _, _, _, _ = env.step(None)
        batch_rest_poses = obs["extra"]["actor_pose"][batch_mask]
        rest_poses.append(batch_rest_poses)

    rest_poses = torch.cat(rest_poses, dim=0)
    rest_poses[..., 3:] = standardize_quaternion(rest_poses[..., 3:])

    # Move scene_desc_poses to the same device as rest_poses
    scene_desc_poses = scene_desc_poses.to(rest_poses.device)

    # Check if objects kept their poses (position and orientation should be stable)
    valid_pos = torch.norm(scene_desc_poses[..., :3] - rest_poses[..., :3], dim=-1) < 5e-3
    valid_quat = torch.norm(scene_desc_poses[..., 3:] - rest_poses[..., 3:], dim=-1) < 5e-2
    valid_obj = torch.logical_and(valid_pos, valid_quat).cpu()
    logger.info(f"Found {torch.logical_not(valid_obj).sum()} unstable objects out of {len(valid_obj)}")

    valid_scene_descriptions = [scene_desc for scene_desc, valid in zip(scene_desc_candidates, valid_obj) if valid]
    valid_scene_descriptions = filter_invalid_scene_descriptions(valid_scene_descriptions, cfg)
    if cfg.enable_object_rebalancing:
        valid_scene_descriptions = rebalance_scene_descriptions_based_on_object_counts(
            valid_scene_descriptions, cfg, mode="canonical_scene_id"
        )
    if cfg.enable_height_rebalancing:
        valid_scene_descriptions = rebalance_scene_descriptions_based_on_heights(valid_scene_descriptions, cfg)

    rng = np.random.RandomState(42)
    max_num_scenes_for_rank = cfg.max_num_scenes // cfg.world_size if cfg.world_size > 1 else cfg.max_num_scenes
    if len(valid_scene_descriptions) > max_num_scenes_for_rank:
        logger.info(f"Subsampling {len(valid_scene_descriptions)} to {max_num_scenes_for_rank}...")
        valid_scene_descriptions = rng.choice(valid_scene_descriptions, max_num_scenes_for_rank, replace=False)
    inspect_scene_descriptions(valid_scene_descriptions, mode="canonical_scene_id")

    save_valid_scene_descriptions(valid_scene_descriptions, cfg)


def simulate_and_validate_scene_descriptions(cfg: Config):
    """Dispatcher function to call the appropriate workflow based on source_type."""
    if cfg.source_type == "meshes":
        logger.info("Generating canonical tabletop scenes from object meshes...")
        generate_canonical_from_meshes(cfg)
    elif cfg.source_type == "canonical":
        logger.info("Generating non-canonical tabletop scenes from canonical scene descriptions...")
        generate_noncanonical_from_canonical(cfg)
    else:
        raise ValueError(f"Invalid source_type: {cfg.source_type}. Must be 'meshes' or 'canonical'.")


def inspect_scene_descriptions(
    scene_desc_candidates: list[SceneDescription],
    mode: Literal["mesh_path", "floating_scene_id", "canonical_scene_id"] = "mesh_path",
):
    """Print statistics for object counts and object height distribution in scene descriptions."""
    logger.info(f"Stats for {len(scene_desc_candidates)} scene descriptions based on {mode} counts...")
    obj_counts: dict[str, int] = {}
    for scene in scene_desc_candidates:
        if mode == "mesh_path":
            key = scene.meshes["object"].collision_mesh_path
        elif mode == "floating_scene_id":
            key = scene.floating_scene_id
        elif mode == "canonical_scene_id":
            key = scene.canonical_tabletop_scene_id
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if key is None:
            key = "None"
        obj_counts[key] = obj_counts.get(key, 0) + 1

    counts = list(obj_counts.values())
    if counts:
        mean_count = float(np.mean(counts))
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
    else:
        mean_count = 0.0
        min_count = 0
        max_count = 0

    logger.info(f"Object distribution: total unique objects = {len(obj_counts)}")
    logger.info(f"Scene counts per object: mean = {mean_count:.2f}, min = {min_count}, max = {max_count}")

    scene_count_group = Counter(counts)
    logger.info("Grouped object distribution by scene count:")
    max_bar_len = 40
    max_group_count = max(scene_count_group.values()) if scene_count_group else 1
    for sc in sorted(scene_count_group.keys()):
        group_count = scene_count_group[sc]
        bar = "*" * int((group_count / max_group_count) * max_bar_len) if max_group_count > 0 else ""
        logger.info(f"Scene count {sc}: {group_count} objects {bar}")

    heights = []
    for scene in scene_desc_candidates:
        bounds = scene.meshes["object"].get_bounds()
        z_min, z_max = bounds[2]
        heights.append(z_max - z_min)
    heights_arr = np.array(heights)
    if heights_arr.size > 0:
        mean_height = float(np.mean(heights_arr))
        min_height = float(np.min(heights_arr))
        max_height = float(np.max(heights_arr))
    else:
        mean_height = 0.0
        min_height = 0.0
        max_height = 0.0

    logger.info("Height distribution (object height):")
    logger.info(f"Mean = {mean_height:.4f}, Min = {min_height:.4f}, Max = {max_height:.4f}")
    bins = 10
    hist, bin_edges = np.histogram(heights_arr, bins=bins)
    logger.info("Bar distribution for height:")
    max_hist = int(np.max(hist)) if hist.size > 0 else 1
    for i in range(bins):
        bin_range = f"[{bin_edges[i]:.4f}, {bin_edges[i + 1]:.4f})"
        bar = "*" * int((hist[i] / max_hist) * max_bar_len) if max_hist > 0 else ""
        logger.info(f"{bin_range}: {hist[i]} {bar}")


CONFIGS: dict[str, Config] = {
    # ---------------------------------------------- v0.3 ----------------------------------------------
    # floating-dependency generation presets are removed, but their canonical msgpacks on disk still work below
    "dex1b_v0.3_tabletop_mini_train": (
        dex1b_v0_3_tabletop_mini_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_mini_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=5000,
            max_scene_count_per_object=10,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            height_rebalancing_bounds=[
                (0.02, 0.05),
                (0.05, 0.09),
                (0.09, 0.15),
                (0.15, 0.25),
            ],
        )
    ),
    "dex1b_v0.3_tabletop_mini_test": replace(
        dex1b_v0_3_tabletop_mini_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_mini.msgpack"),
        ],
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_mini_test.msgpack"),
        max_num_scenes=500,
        split="test",
    ),
    "dex1b_v0.3_tabletop_small_train": (
        dex1b_v0_3_tabletop_small_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_small.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_small_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=50000,
            max_scene_count_per_object=10,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            height_rebalancing_bounds=[
                (0.02, 0.05),
                (0.05, 0.09),
                (0.09, 0.15),
                (0.15, 0.25),
            ],
        )
    ),
    "dex1b_v0.3_tabletop_small_test": replace(
        dex1b_v0_3_tabletop_small_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_small.msgpack"),
        ],
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_small_test.msgpack"),
        max_num_scenes=2000,
        split="test",
    ),
    # ---------------------------------------------- v0.4 ----------------------------------------------
    "dex1b_v0.4_tabletop_mini_train": (
        dex1b_v0_4_tabletop_mini_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_mini_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=5000,
            max_scene_count_per_object=10,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            height_rebalancing_bounds=[
                (0.02, 0.05),
                (0.05, 0.09),
                (0.09, 0.15),
                (0.15, 0.25),
            ],
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4_tabletop_mini_test": replace(
        dex1b_v0_4_tabletop_mini_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini.msgpack"),
        ],
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_mini_test.msgpack"),
        max_num_scenes=500,
        split="test",
    ),
    "dex1b_v0.4_small_world_refined_tabletop_mini_train": (
        dex1b_v0_4_small_world_refined_tabletop_mini_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/small_world_refined_tabletop_mini_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=5000,
            max_scene_count_per_object=5,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_table_pose_randomization=True,
            table_x_range=(0.35, 0.6),
            table_y_range=(-0.4, 0.3),
            table_height_range=(0.5, 1.0),
            # valid_xy_bounds (relative to per-scene table xy; see filter_invalid_scene_descriptions) left at default
            enable_height_rebalancing=False,
            enable_object_rebalancing=True,
            table_size=(0.25, 0.3, 0.02),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4_small_world_refined_tabletop_mini_test": replace(
        dex1b_v0_4_small_world_refined_tabletop_mini_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/small_world_refined_tabletop_mini_test.msgpack"),
        max_num_scenes=500,
    ),
    # ---------------------------------------------- v0.4 dev ----------------------------------------------
    # YCB-only dev variant: fastest iteration loop, ~30 objects shared between train/test, split via subsampling
    "dex1b_v0.4_dev_canonical_tabletop_train": (
        dex1b_v0_4_dev_canonical_tabletop_train := Config(
            source_type="meshes",
            mesh_sources=["dexdiffuser_v2"],
            dexdiffuser_sources=["ycb"],
            scales=[1.0],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev_train.msgpack"),
            split="train",
            canonicalized_pose=True,
            num_envs=512,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4_dev_canonical_tabletop_test": replace(
        dex1b_v0_4_dev_canonical_tabletop_train,
        split="test",
        max_num_scenes=30,
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev_test.msgpack"),
    ),
    "dex1b_v0.4_dev_tabletop_train": (
        dex1b_v0_4_dev_tabletop_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_dev_train.msgpack"),
            split="train",
            num_envs=1024,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=100,
            max_scene_count_per_object=5,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            height_rebalancing_bounds=[
                (0.02, 0.05),
                (0.05, 0.09),
                (0.09, 0.15),
                (0.15, 0.25),
            ],
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4_dev_tabletop_test": replace(
        dex1b_v0_4_dev_tabletop_train,
        split="test",
        max_num_scenes=30,
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_dev_test.msgpack"),
    ),
    # ---------------------------------------------- v0.4.1 ----------------------------------------------
    # small tier: dexgraspnet-only (all train objects x 5 scales), no objaverse.
    "dex1b_v0.4.1_canonical_tabletop_small_train": (
        dex1b_v0_4_1_canonical_tabletop_small_train := Config(
            source_type="meshes",
            mesh_sources=["dexgraspnet"],
            scales=[0.06, 0.08, 0.1, 0.12, 0.15],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small_train.msgpack"),
            split="train",
            canonicalized_pose=True,
            num_envs=4096,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4.1_canonical_tabletop_small_test": replace(
        dex1b_v0_4_1_canonical_tabletop_small_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small_test.msgpack"),
    ),
    "dex1b_v0.4.1_small_world_tabletop_mini_train": (
        dex1b_v0_4_1_small_world_tabletop_mini_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_mini_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=60,
            max_num_scenes=10000,
            max_scene_count_per_object=10,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_table_pose_randomization=True,
            table_x_range=(0.3, 0.7),
            table_y_range=(-0.45, 0.35),
            table_height_range=(0.3, 1.1),
            enable_height_rebalancing=False,
            enable_object_rebalancing=True,
            table_size=(0.25, 0.3, 0.02),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4.1_small_world_tabletop_mini_test": replace(
        dex1b_v0_4_1_small_world_tabletop_mini_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_mini_test.msgpack"),
        max_num_scenes=1000,
    ),
    "dex1b_v0.4.1_small_world_tabletop_small_train": (
        dex1b_v0_4_1_small_world_tabletop_small_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_small_train.msgpack"),
            split="train",
            num_envs=4096,
            # canonical_small already has ~50k scenes, so few variants/scene suffice (mini needs 30+ from ~1.7k)
            num_candidates_per_canonical_scene=2,
            max_num_scenes=50000,
            max_scene_count_per_object=2,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_table_pose_randomization=True,
            table_x_range=(0.3, 0.7),
            table_y_range=(-0.45, 0.35),
            table_height_range=(0.3, 1.1),
            enable_height_rebalancing=False,
            enable_object_rebalancing=True,
            table_size=(0.25, 0.3, 0.02),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.4.1_small_world_tabletop_small_test": replace(
        dex1b_v0_4_1_small_world_tabletop_small_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_small_test.msgpack"),
        max_num_scenes=2000,
    ),
    # ---------------------------------------------- v0.5.0 ----------------------------------------------
    # Self-contained: canonicals regenerated from meshes (stage 1) into v0.5.0/, then tabletop rand (stage 2). Families:
    # - tabletop: fixed table at origin xy, pose-rand only jitters table height
    # - wbc_tabletop: WBC, per-scene table pose pulled closer to the robot
    # --- stage 1: canonical generation (source_type="meshes") ---
    "dex1b_v0.5.0_canonical_tabletop_dev_train": (
        dex1b_v0_5_0_canonical_tabletop_dev_train := Config(
            source_type="meshes",
            mesh_sources=["dexdiffuser_v2"],
            dexdiffuser_sources=["ycb"],
            scales=[1.0],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev_train.msgpack"),
            split="train",
            canonicalized_pose=True,
            num_envs=512,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.5.0_canonical_tabletop_dev_test": replace(
        dex1b_v0_5_0_canonical_tabletop_dev_train,
        split="test",
        max_num_scenes=30,
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev_test.msgpack"),
    ),
    "dex1b_v0.5.0_canonical_tabletop_mini_train": (
        dex1b_v0_5_0_canonical_tabletop_mini_train := Config(
            source_type="meshes",
            mesh_sources=["dexgraspnet"],
            scales=[0.06, 0.08, 0.1, 0.12, 0.15],
            max_num_objects=684,
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_train.msgpack"),
            split="train",
            canonicalized_pose=True,
            num_envs=4096,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.5.0_canonical_tabletop_mini_test": (
        dex1b_v0_5_0_canonical_tabletop_mini_test := Config(
            source_type="meshes",
            mesh_sources=["dexdiffuser_v2"],
            dexdiffuser_sources=["egad", "kit", "contactdb", "ycb"],
            scales=[1.0],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_test.msgpack"),
            split="test",
            canonicalized_pose=True,
            num_envs=4096,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    # dexgraspnet test-split (x5 scales, all test objects) for the normal mini test 300/200 split.
    "dex1b_v0.5.0_canonical_tabletop_mini_dexgraspnet_test": (
        dex1b_v0_5_0_canonical_tabletop_mini_dexgraspnet_test := replace(
            dex1b_v0_5_0_canonical_tabletop_mini_train,
            max_num_objects=None,
            split="test",
            output_path=str(
                DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_dexgraspnet_test.msgpack"
            ),
        )
    ),
    "dex1b_v0.5.0_canonical_tabletop_small_train": (
        dex1b_v0_5_0_canonical_tabletop_small_train := replace(
            dex1b_v0_5_0_canonical_tabletop_mini_train,
            max_num_objects=None,
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small_train.msgpack"),
        )
    ),
    "dex1b_v0.5.0_canonical_tabletop_small_test": (
        dex1b_v0_5_0_canonical_tabletop_small_test := Config(
            source_type="meshes",
            mesh_sources=["dexgraspnet", "dexdiffuser_v2"],
            dexdiffuser_sources=["egad", "kit", "contactdb", "ycb"],
            scales=[0.06, 0.08, 0.1, 0.12, 0.15],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small_test.msgpack"),
            split="test",
            canonicalized_pose=True,
            num_envs=4096,
            orientations=["none", "x90", "y90"],
            enable_height_rebalancing=False,
            enable_object_rebalancing=False,
            table_size=(0.3, 0.3, 0.1),
            table_pos=(0.0, 0.0, -0.05),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    # --- stage 2a: normal tabletop (fixed table at origin xy, +-5cm table-height jitter) ---
    "dex1b_v0.5.0_tabletop_small_train": (
        dex1b_v0_5_0_tabletop_small_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_small_train.msgpack"),
            split="train",
            num_envs=4096,
            # normal family loses ~55% pre-sim to the jitter filter and height rebalancing, so over-generate to hit 50k
            num_candidates_per_canonical_scene=7,
            max_num_scenes=50000,
            max_scene_count_per_object=6,
            enable_xy_translation_randomization=True,
            xy_translation_randomization_bounds=((-0.2, 0.2), (-0.2, 0.2)),
            enable_z_rotation_randomization=True,
            valid_xy_bounds=((-0.2, 0.2), (-0.2, 0.2)),
            enable_table_pose_randomization=True,
            table_x_range=(0.0, 0.0),
            table_y_range=(0.0, 0.0),
            table_height_range=(-0.05, 0.05),
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            table_size=(0.5, 0.5, 0.2),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.5.0_tabletop_small_test": replace(
        dex1b_v0_5_0_tabletop_small_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_small_test.msgpack"),
        max_num_scenes=2000,
    ),
    "dex1b_v0.5.0_tabletop_mini_train": (
        dex1b_v0_5_0_tabletop_mini_train := replace(
            dex1b_v0_5_0_tabletop_small_train,
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_train.msgpack"),
            num_candidates_per_canonical_scene=30,
            max_num_scenes=5000,
            max_scene_count_per_object=10,
        )
    ),
    "dex1b_v0.5.0_tabletop_mini_test": replace(
        dex1b_v0_5_0_tabletop_mini_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test.msgpack"),
        max_num_scenes=1000,
    ),
    # dexmate family: same mini canonicals, widened table height/region/size for the vega+sharpa humanoid
    "dex1b_v0.5.0_dexmate_tabletop_mini_train": (
        dex1b_v0_5_0_dexmate_tabletop_mini_train := replace(
            dex1b_v0_5_0_tabletop_mini_train,
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_train.msgpack"),
            table_height_range=(-0.15, 0.15),
            xy_translation_randomization_bounds=((-0.25, 0.25), (-0.25, 0.25)),
            valid_xy_bounds=((-0.25, 0.25), (-0.25, 0.25)),
            table_size=(0.6, 0.6, 0.2),
        )
    ),
    "dex1b_v0.5.0_dexmate_tabletop_mini_test": replace(
        dex1b_v0_5_0_dexmate_tabletop_mini_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test.msgpack"),
        max_num_scenes=1000,
    ),
    "dex1b_v0.5.0_dexmate_tabletop_mini_test_dexgraspnet": replace(
        dex1b_v0_5_0_dexmate_tabletop_mini_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_dexgraspnet_test.msgpack"),
        ],
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test_dexgraspnet.msgpack"),
        num_candidates_per_canonical_scene=2,
        max_num_scenes=1000,
    ),
    # dexgraspnet portion of the mini test (300 + dexdiffuser's 200 = 500); ~9.8k canonical scenes clear the cap easily
    "dex1b_v0.5.0_tabletop_mini_test_dexgraspnet": replace(
        dex1b_v0_5_0_tabletop_mini_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_dexgraspnet_test.msgpack"),
        ],
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test_dexgraspnet.msgpack"),
        num_candidates_per_canonical_scene=2,
        max_num_scenes=1000,
    ),
    "dex1b_v0.5.0_tabletop_dev_train": (
        dex1b_v0_5_0_tabletop_dev_train := replace(
            dex1b_v0_5_0_tabletop_small_train,
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_dev_train.msgpack"),
            num_envs=1024,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=100,
            max_scene_count_per_object=5,
            enable_height_rebalancing=False,
        )
    ),
    "dex1b_v0.5.0_tabletop_dev_test": replace(
        dex1b_v0_5_0_tabletop_dev_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_dev_test.msgpack"),
        max_num_scenes=30,
    ),
    # --- stage 2b: WBC tabletop (per-scene table pose, table closer to robot x 0.3-0.5) ---
    "dex1b_v0.5.0_wbc_tabletop_small_train": (
        dex1b_v0_5_0_wbc_tabletop_small_train := Config(
            source_type="canonical",
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_small_train.msgpack"),
            split="train",
            num_envs=4096,
            num_candidates_per_canonical_scene=7,
            max_num_scenes=50000,
            max_scene_count_per_object=6,
            enable_xy_translation_randomization=True,
            enable_z_rotation_randomization=True,
            enable_table_pose_randomization=True,
            table_x_range=(0.35, 0.5),
            table_y_range=(-0.45, 0.35),
            table_height_range=(0.3, 1.1),
            enable_height_rebalancing=True,
            enable_object_rebalancing=True,
            table_size=(0.35, 0.35, 0.2),
            table_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ),
    "dex1b_v0.5.0_wbc_tabletop_small_test": replace(
        dex1b_v0_5_0_wbc_tabletop_small_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_small_test.msgpack"),
        max_num_scenes=2000,
    ),
    "dex1b_v0.5.0_wbc_tabletop_mini_train": (
        dex1b_v0_5_0_wbc_tabletop_mini_train := replace(
            dex1b_v0_5_0_wbc_tabletop_small_train,
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_train.msgpack"),
            num_candidates_per_canonical_scene=30,
            max_num_scenes=5000,
            max_scene_count_per_object=10,
        )
    ),
    "dex1b_v0.5.0_wbc_tabletop_mini_test": replace(
        dex1b_v0_5_0_wbc_tabletop_mini_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test.msgpack"),
        max_num_scenes=1000,
    ),
    # dexgraspnet portion of the WBC mini test (merged with the dexdiffuser test, 300/200, = 500).
    "dex1b_v0.5.0_wbc_tabletop_mini_test_dexgraspnet": replace(
        dex1b_v0_5_0_wbc_tabletop_mini_train,
        canonical_scene_description_paths=[
            str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_dexgraspnet_test.msgpack"),
        ],
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test_dexgraspnet.msgpack"),
        num_candidates_per_canonical_scene=2,
        max_num_scenes=1000,
    ),
    "dex1b_v0.5.0_wbc_tabletop_dev_train": (
        dex1b_v0_5_0_wbc_tabletop_dev_train := replace(
            dex1b_v0_5_0_wbc_tabletop_small_train,
            canonical_scene_description_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev.msgpack"),
            ],
            output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_dev_train.msgpack"),
            num_envs=1024,
            num_candidates_per_canonical_scene=30,
            max_num_scenes=100,
            max_scene_count_per_object=5,
            enable_height_rebalancing=False,
        )
    ),
    "dex1b_v0.5.0_wbc_tabletop_dev_test": replace(
        dex1b_v0_5_0_wbc_tabletop_dev_train,
        split="test",
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_dev_test.msgpack"),
        max_num_scenes=30,
    ),
}


if __name__ == "__main__":
    seed_everything(42)
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    simulate_and_validate_scene_descriptions(cfg)
