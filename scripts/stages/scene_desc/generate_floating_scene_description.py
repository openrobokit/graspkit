"""
Generate single floating object scene descriptions.

Usage:
python scripts/stages/data_process/generate_scene_description_floating.py dexdiffuser_floating
"""

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import tyro
from rich.progress import track

from graspkit.assets.objects import dexgraspnet_official_splits
from graspkit.assets.objects.discovery import (
    find_dexdiffuser_assets_v1,
    find_dexdiffuser_assets_v2,
    find_dexgraspnet_assets,
    find_objaverse_dex1b_assets,
    find_robocasa_assets,
)
from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from graspkit.scene.scene_description import MeshDescription, SceneDescription

logger = logging.getLogger("graspkit")


@dataclass
class Config:
    data_sources: list[
        Literal[
            "dexdiffuser",
            "dexdiffuser_v2",
            "dexgraspnet_eval",
            "dexgraspnet_train",
            "objaverse_dex1b",
            "robocasa",
        ]
    ] = field(default_factory=lambda: ["dexdiffuser"])
    """Data sources to use."""
    output_path: str = str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack")
    """Path to save the scene description."""

    num_instances_per_object: int = 1
    """Number of instances to generate per object."""
    max_num_scenes: int = 1_000_000
    """Maximum number of scenes to generate."""

    enable_scale_randomization: bool = False
    """Whether to enable scale randomization."""
    scale_randomization_bounds: tuple[float, float] = (0.05, 0.15)
    """Bounds of the scale randomization."""
    scales: list[float] = field(default_factory=lambda: [1.0])
    """Scales to apply to the objects."""

    # Dex Diffuser (egad_eval, kit, multidex); object dirs provisioned on demand via graspkit.assets.objects imports
    dexdiffuser_sources: list[str] | None = None
    """Subset of dexdiffuser_v2 source directories to scan (None = all: egad, kit, contactdb, ycb)."""

    # Dexgraspnet
    dexgraspnet_split_dir: str = str(dexgraspnet_official_splits.DIR)
    """Path to the Dexgraspnet split directory."""

    # Robocasa (no HF archive; extract locally)
    robocasa_data_dir: str = str(PROJECT_ROOT / "data/object_data/robocasa/processed")
    """Path to the Robocasa object data directory."""

    def __post_init__(self) -> None:
        if not self.enable_scale_randomization:
            if len(self.scales) != self.num_instances_per_object:
                raise ValueError(
                    f"When enable_scale_randomization is False, len(scales) ({len(self.scales)}) "
                    f"must equal num_instances_per_object ({self.num_instances_per_object})"
                )


def inspect_floating_scene_descriptions(output_path: str):
    """Inspect and report statistics of floating object meshes."""
    assert os.path.exists(output_path), f"Output path does not exist: {output_path}"
    scene_descs = SceneDescription.load_scene_descriptions_msgpack(output_path)
    logger.info(f"Loaded {len(scene_descs)} scene descriptions")

    bounds = []
    for scene_desc in scene_descs.values():
        mesh_desc = scene_desc.meshes[scene_desc.target_actor_id]
        bounds.append(mesh_desc.bounds)

    bounds_arr = np.array(bounds)  # Shape: (N, 3, 2) where each row is ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    centers_arr = bounds_arr.mean(axis=2)  # Shape: (N, 3)
    sizes_arr = bounds_arr[:, :, 1] - bounds_arr[:, :, 0]  # Shape: (N, 3)

    logger.info(f"\nCenter (x, y, z) - {len(bounds)} instances:")
    logger.info(f"  Min:  {centers_arr.min(axis=0)}")
    logger.info(f"  Max:  {centers_arr.max(axis=0)}")
    logger.info(f"  Mean: {centers_arr.mean(axis=0)}")

    logger.info("\nSize (width, depth, height):")
    logger.info(f"  Min:  {sizes_arr.min(axis=0)}")
    logger.info(f"  Max:  {sizes_arr.max(axis=0)}")
    logger.info(f"  Mean: {sizes_arr.mean(axis=0)}")


def generate_floating_scene_descriptions(cfg: Config):
    """Generate and save floating scene descriptions."""
    rng = np.random.RandomState(42)
    scene_descs: list[SceneDescription] = []
    for data_source in cfg.data_sources:
        if data_source == "robocasa":
            model_names, mesh_paths, visual_mesh_paths = find_robocasa_assets(cfg.robocasa_data_dir)
        elif data_source == "dexdiffuser":
            from graspkit.assets.objects import dex_diffuser

            model_names, mesh_paths, visual_mesh_paths = find_dexdiffuser_assets_v1(str(dex_diffuser.DATA_DIR))
        elif data_source == "dexdiffuser_v2":
            from graspkit.assets.objects import dex_diffuser_v2

            model_names, mesh_paths, visual_mesh_paths = find_dexdiffuser_assets_v2(
                str(dex_diffuser_v2.DATA_DIR), sources=cfg.dexdiffuser_sources
            )
        elif data_source in ("dexgraspnet_eval", "dexgraspnet_train"):
            from graspkit.assets.objects import dexgraspnet

            split = "test" if data_source == "dexgraspnet_eval" else "train"
            model_names, mesh_paths = find_dexgraspnet_assets(
                str(dexgraspnet.DATA_DIR), cfg.dexgraspnet_split_dir, split=split
            )
            visual_mesh_paths = None
        elif data_source == "objaverse_dex1b":
            from graspkit.assets.objects import objaverse_dex1b

            model_names, mesh_paths, visual_mesh_paths = find_objaverse_dex1b_assets(
                str(objaverse_dex1b.DATA_DIR / "processed")
            )
        else:
            raise ValueError(f"Invalid data source: {data_source}")
        logger.info(f"Found {len(model_names)} {data_source} models for {data_source}")

        for model_idx in track(
            range(len(model_names)),
            description=f"Initializing for {data_source}...",
            # disable=True,
        ):
            model_name = model_names[model_idx]
            coll_mesh_path = mesh_paths[model_idx]
            visual_mesh_path = visual_mesh_paths[model_idx] if visual_mesh_paths is not None else None

            if not cfg.enable_scale_randomization:
                scales = cfg.scales
            else:
                scales = rng.uniform(
                    cfg.scale_randomization_bounds[0],
                    cfg.scale_randomization_bounds[1],
                    size=cfg.num_instances_per_object,
                ).tolist()

            for instance_idx in range(cfg.num_instances_per_object):
                scale = scales[instance_idx % len(scales)]

                obj_desc = MeshDescription(
                    actor_id="object",
                    collision_mesh_path=os.path.relpath(coll_mesh_path, PROJECT_ROOT),
                    visual_mesh_path=os.path.relpath(visual_mesh_path, PROJECT_ROOT)
                    if visual_mesh_path is not None
                    else None,
                    scales=(scale, scale, scale),
                    position=(0.0, 0.0, 0.0),
                    quaternion=(1.0, 0.0, 0.0, 0.0),
                )
                bounds = obj_desc.get_bounds()
                obj_desc = replace(obj_desc, bounds=bounds)

                scene_desc = SceneDescription(
                    scene_id=f"{data_source}_{model_name}_inst_{instance_idx}_scale_{scale:.3f}_floating",
                    boxes={},
                    meshes={obj_desc.actor_id: obj_desc},
                    target_actor_id=obj_desc.actor_id,
                )
                scene_descs.append(scene_desc)
    logger.info(f"Generated {len(scene_descs)} floating scene descriptions")

    # Subsample if needed
    if len(scene_descs) > cfg.max_num_scenes:
        scene_descs = rng.choice(scene_descs, cfg.max_num_scenes, replace=False).tolist()
        logger.info(f"Subsampled to {len(scene_descs)} scene descriptions")

    output_path = cfg.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    SceneDescription.save_scene_descriptions_msgpack(
        {scene_desc.scene_id: scene_desc for scene_desc in scene_descs}, msgpack_path=output_path
    )
    logger.info(f"Saved {len(scene_descs)} floating scene descriptions to {output_path}")

    inspect_floating_scene_descriptions(output_path)


CONFIGS: dict[str, Config] = {
    "dexdiffuser_floating": Config(
        data_sources=["dexdiffuser"],
        num_instances_per_object=1,
        scales=[1.0],  # follow dexdiffuser to use scale 1.0
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack"),
    ),
    "dexgraspnet_floating_eval": Config(
        data_sources=["dexgraspnet_eval"],
        num_instances_per_object=5,
        scales=[0.06, 0.08, 0.1, 0.12, 0.15],  # follow dexgraspnet to use 5 fixed scales
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/dexgraspnet/floating_eval.msgpack"),
    ),
    "dexgraspnet_floating_train": Config(
        data_sources=["dexgraspnet_train"],
        num_instances_per_object=5,
        scales=[0.06, 0.08, 0.1, 0.12, 0.15],  # follow dexgraspnet to use 5 fixed scales
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"),
    ),
    "objaverse_dex1b_floating": Config(
        data_sources=["objaverse_dex1b"],
        num_instances_per_object=1,
        enable_scale_randomization=True,
        scale_randomization_bounds=(0.05, 0.15),
        output_path=str(DATA_ROOT / "scene_description/objaverse_dex1b/floating.msgpack"),
    ),
    "robocasa_floating": Config(
        data_sources=["robocasa"],
        num_instances_per_object=1,
        scales=[1.0],
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/robocasa/floating.msgpack"),
    ),
    "dexdiffuser_v2_floating": Config(
        data_sources=["dexdiffuser_v2"],
        num_instances_per_object=1,
        scales=[1.0],
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating.msgpack"),
    ),
    "dexdiffuser_v2_ycb_floating": Config(
        data_sources=["dexdiffuser_v2"],
        dexdiffuser_sources=["ycb"],
        num_instances_per_object=1,
        scales=[1.0],
        enable_scale_randomization=False,
        output_path=str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating_ycb.msgpack"),
    ),
}


if __name__ == "__main__":
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    generate_floating_scene_descriptions(cfg)
