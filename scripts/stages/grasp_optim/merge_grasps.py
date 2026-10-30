"""
Merge per-scene grasp files into a single HDF5 file.

- Takes optimized grasp results from individual scene files and merges them
- Optionally removes source files after merging

Usage:
uv run scripts/stages/grasp_optim/merge_grasps.py data/optim_data/mano_grasps/dex1b_floating_mini
uv run scripts/stages/grasp_optim/merge_grasps.py data/optim_data/xhand_dex1b_floating --rm
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import io
import logging
import os

import fsspec
import h5py
import tyro
from rich.progress import track

from graspkit.utils.io_utils import build_filesystem

logger = logging.getLogger("graspkit")


def merge_grasps(output_dir: str, /, rm: bool = False):
    fs = build_filesystem(output_dir)
    h5_files = fs.glob(output_dir.rstrip("/") + "/*.h5")
    scene_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in h5_files])

    if len(scene_ids) == 0:
        raise ValueError(f"No .h5 files found in {output_dir}")

    dst_file = output_dir.rstrip("/") + ".h5"
    logger.info(f"Merging {len(scene_ids)} per-scene files into {dst_file}")

    empty_count = 0
    non_empty_count = 0

    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as merged_h5:
        for scene_id in track(scene_ids, description="Merging scenes"):
            scene_file_path = os.path.join(output_dir.rstrip("/"), f"{scene_id}.h5")
            with h5py.File(scene_file_path, "r") as scene_group:
                if len(scene_group.keys()) == 0:
                    empty_count += 1
                    continue

                non_empty_count += 1
                merged_group = merged_h5.create_group(scene_id)
                for key in scene_group.keys():
                    merged_group.create_dataset(key, data=scene_group[key][:])

    # Write merged file
    buffer.seek(0)
    with fsspec.open(dst_file, "wb") as f:
        f.write(buffer.read())

    logger.info(f"Merged {non_empty_count} non-empty files into {dst_file}")
    logger.info(f"Skipped {empty_count} empty files")

    # Count total items in merged file
    total_grasp_items = 0
    with fsspec.open(dst_file, "rb") as f:
        with h5py.File(f, "r") as merged_h5:
            for scene_id in merged_h5.keys():
                if "grasp_qpos" in merged_h5[scene_id]:
                    total_grasp_items += merged_h5[scene_id]["grasp_qpos"].shape[0]

    if total_grasp_items > 0:
        logger.info(f"Total grasp items in merged file: {total_grasp_items}")

    # Remove source files if requested
    if rm:
        logger.info("Removing source files...")
        for scene_id in scene_ids:
            scene_file_path = os.path.join(output_dir.rstrip("/"), f"{scene_id}.h5")
            if output_dir.startswith("s3://"):
                fs.rm(scene_file_path)
            else:
                os.remove(scene_file_path)
        logger.info(f"Removed {len(scene_ids)} source files")


if __name__ == "__main__":
    tyro.cli(merge_grasps, use_underscores=True)
