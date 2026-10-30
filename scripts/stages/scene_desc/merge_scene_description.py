"""
Merge multiple scene description files into a single output file.

Usage:
python scripts/stages/scene_desc/merge_scene_description.py dex1b_v0.3_tabletop_mini
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import tyro

from graspkit.constants import DATA_ROOT
from graspkit.scene.scene_description import SceneDescription

logger = logging.getLogger("graspkit")


@dataclass
class SplitConfig:
    input_paths: list[str]
    """Paths to input scene description files to merge for this split."""
    subsample: dict[str, int] = field(default_factory=dict)
    """Optional subsample count per input path. Key is input path, value is number of scenes to subsample."""


@dataclass
class Config:
    train: SplitConfig
    """Configuration for train split."""
    eval: SplitConfig
    """Configuration for eval split."""
    output_path: str
    """Path to save the merged scene descriptions."""
    delete_inputs: bool = False
    """If True, delete input msgpack files after the merged output is saved."""


CONFIGS: dict[str, Config] = {
    # ---------------------------------------------- v0.4 ----------------------------------------------
    "dex1b_v0.4_floating_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"): 1000,
            },
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/floating_mini.msgpack"),
    ),
    "dex1b_v0.4_canonical_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.4_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_mini.msgpack"),
    ),
    "dex1b_v0.4_small_world_refined_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/small_world_refined_tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/small_world_refined_tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/small_world_refined_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.4.1_small_world_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.4.1_canonical_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/canonical_tabletop_small.msgpack"),
    ),
    "dex1b_v0.4.1_small_world_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_small_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_small_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4.1/small_world_tabletop_small.msgpack"),
    ),
    # ---------------------------------------------- v0.5.0 ----------------------------------------------
    # 3 canonical merges (family-agnostic) + 6 tabletop merges (normal + wbc x dev/mini/small).
    "dex1b_v0.5.0_canonical_tabletop_dev": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_dev.msgpack"),
    ),
    "dex1b_v0.5.0_canonical_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.5.0_canonical_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/canonical_tabletop_small.msgpack"),
    ),
    "dex1b_v0.5.0_tabletop_dev": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_dev_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_dev_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_dev.msgpack"),
    ),
    "dex1b_v0.5.0_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_train.msgpack")],
        ),
        # normal mini test = 300 dexgraspnet + 200 dexdiffuser = 500.
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test_dexgraspnet.msgpack"),
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test_dexgraspnet.msgpack"): 300,
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini_test.msgpack"): 200,
            },
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini.msgpack"),
    ),
    "dex1b_v0.5.0_dexmate_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test_dexgraspnet.msgpack"),
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test_dexgraspnet.msgpack"): 300,
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini_test.msgpack"): 200,
            },
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/dexmate_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.5.0_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_small_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_small_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_small.msgpack"),
    ),
    "dex1b_v0.5.0_wbc_tabletop_dev": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_dev_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_dev_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_dev.msgpack"),
    ),
    "dex1b_v0.5.0_wbc_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_train.msgpack")],
        ),
        # WBC mini test = 300 dexgraspnet + 200 dexdiffuser = 500 (same as normal tabletop).
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test_dexgraspnet.msgpack"),
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test_dexgraspnet.msgpack"): 300,
                str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini_test.msgpack"): 200,
            },
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.5.0_wbc_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_small_train.msgpack")],
        ),
        eval=SplitConfig(
            input_paths=[str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_small_test.msgpack")],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.5.0/wbc_tabletop_small.msgpack"),
    ),
    # YCB-only dev variant: train and eval draw from the same YCB floating pool with different subsample counts.
    "dex1b_v0.4_dev_floating": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating_ycb.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating_ycb.msgpack"): 50,
            },
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating_ycb.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexdiffuser_v2/floating_ycb.msgpack"): 20,
            },
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/floating_dev.msgpack"),
    ),
    "dex1b_v0.4_dev_canonical_tabletop": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/canonical_tabletop_dev.msgpack"),
        delete_inputs=True,
    ),
    "dex1b_v0.4_dev_tabletop": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_dev_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_dev_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.4/tabletop_dev.msgpack"),
        delete_inputs=True,
    ),
    # ---------------------------------------------- v0.3 ----------------------------------------------
    "dex1b_v0.3_floating": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"),
                str(DATA_ROOT / "scene_description/objaverse_dex1b/floating.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack"),
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_eval.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/floating.msgpack"),
    ),
    "dex1b_v0.3_floating_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"): 1000,
            },
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/floating_mini.msgpack"),
    ),
    "dex1b_v0.3_floating_small": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"),
                str(DATA_ROOT / "scene_description/objaverse_dex1b/floating.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_train.msgpack"): 2000,
                str(DATA_ROOT / "scene_description/objaverse_dex1b/floating.msgpack"): 8000,
            },
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack"),
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_eval.msgpack"),
            ],
            subsample={
                str(DATA_ROOT / "scene_description/dexdiffuser/floating.msgpack"): 100,
                str(DATA_ROOT / "scene_description/dexgraspnet/floating_eval.msgpack"): 900,
            },
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/floating_small.msgpack"),
    ),
    "dex1b_v0.3_canonical_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.3_canonical_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_small_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_small_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/canonical_tabletop_small.msgpack"),
    ),
    "dex1b_v0.3_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_mini.msgpack"),
    ),
    "dex1b_v0.3_tabletop_small": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_small_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_small_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/tabletop_small.msgpack"),
    ),
    "dex1b_v0.3_small_world_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3_small_table/small_world_tabletop_mini_train.msgpack"),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(DATA_ROOT / "scene_description/dex1b/v0.3_small_table/small_world_tabletop_mini_test.msgpack"),
            ],
        ),
        output_path=str(DATA_ROOT / "scene_description/dex1b/v0.3_small_table/small_world_tabletop_mini.msgpack"),
    ),
    "dex1b_v0.3_small_world_refined_tabletop_mini": Config(
        train=SplitConfig(
            input_paths=[
                str(
                    DATA_ROOT
                    / "scene_description/dex1b/v0.3_small_table/small_world_refined_tabletop_mini_train.msgpack"
                ),
            ],
        ),
        eval=SplitConfig(
            input_paths=[
                str(
                    DATA_ROOT
                    / "scene_description/dex1b/v0.3_small_table/small_world_refined_tabletop_mini_test.msgpack"
                ),
            ],
        ),
        output_path=str(
            DATA_ROOT / "scene_description/dex1b/v0.3_small_table/small_world_refined_tabletop_mini.msgpack"
        ),
    ),
}


def merge_scene_descriptions(cfg: Config):
    """Merge multiple scene description files into train and eval splits."""
    splits: dict[str, dict[str, SceneDescription]] = {}

    for split_name, split_cfg in [("train", cfg.train), ("test", cfg.eval)]:
        logger.info(f"Merging {split_name} split...")
        merged_scenes: dict[str, SceneDescription] = {}

        for input_path in split_cfg.input_paths:
            if not os.path.exists(input_path):
                logger.warning(f"Input path does not exist, skipping: {input_path}")
                continue

            logger.info(f"Loading scene descriptions from {input_path}")
            scene_descs = SceneDescription.load_scene_descriptions_msgpack(input_path)
            logger.info(f"Loaded {len(scene_descs)} scene descriptions")

            # Subsample if specified
            if input_path in split_cfg.subsample:
                subsample_count = split_cfg.subsample[input_path]
                scene_ids = list(scene_descs.keys())
                rng = np.random.RandomState(42)
                sampled_ids = rng.choice(scene_ids, min(subsample_count, len(scene_ids)), replace=False).tolist()
                scene_descs = {sid: scene_descs[sid] for sid in sampled_ids}
                logger.info(f"Subsampled to {len(scene_descs)} scene descriptions")

            for scene_id, scene_desc in scene_descs.items():
                if scene_id in merged_scenes:
                    logger.warning(f"Duplicate scene_id found: {scene_id}, overwriting...")
                merged_scenes[scene_id] = scene_desc

        logger.info(f"Merged total: {len(merged_scenes)} scene descriptions for {split_name}")
        splits[split_name] = merged_scenes

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    SceneDescription.save_scene_descriptions_msgpack(splits, msgpack_path=cfg.output_path)
    logger.info(f"Saved {len(splits['train'])} train and {len(splits['test'])} test scenes to {cfg.output_path}")

    if cfg.delete_inputs:
        seen: set[str] = set()
        for split_cfg in (cfg.train, cfg.eval):
            for input_path in split_cfg.input_paths:
                if input_path == cfg.output_path or input_path in seen or not os.path.exists(input_path):
                    continue
                seen.add(input_path)
                os.remove(input_path)
                logger.info(f"Deleted input {input_path}")


if __name__ == "__main__":
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    merge_scene_descriptions(cfg)
