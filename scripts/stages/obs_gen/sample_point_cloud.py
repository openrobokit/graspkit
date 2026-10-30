"""
Usage:
uv run scripts/stages/obs_gen/sample_point_cloud.py dex1b_floating_mini
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
import os
from dataclasses import dataclass
from typing import Dict

import fpsample
import h5py
import numpy as np
import torch
from robokit.xform.numpy import pose7_to_tf_mat
import tyro
from rich.progress import track

from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import get_cached_trimesh_from_file
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.stage_utils import write_completion_marker

logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

SAMPLE_POINT_CLOUD_COMPLETE_MARKER = ".sample_point_cloud_complete"


@dataclass
class Config:
    scene_description_path: str = str(DATA_ROOT / "scene_description/ycb/1_pose_1_scale.json")
    """Path to the scene description"""
    output_path: str = str(PROJECT_ROOT / "data/object_data/full_pointcloud/ycb_1_pose_1_scale.h5")
    """Output path"""
    save_as_separate_files: bool = False
    """Whether to save the point cloud as separate files"""
    num_points: int = 8192
    """Number of points to sample from the point cloud"""
    enable_fps: bool = False
    """Whether to enable FPS sampling"""


def sample_point_cloud(cfg: Config):
    scene_descriptions: Dict[str, SceneDescription] = SceneDescription.load_scene_descriptions_msgpack(
        cfg.scene_description_path
    )
    if cfg.save_as_separate_files:
        os.makedirs(cfg.output_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    def get_pcd(scene_description: SceneDescription):
        target_object = scene_description.meshes[scene_description.target_actor_id]
        coll_mesh = get_cached_trimesh_from_file(target_object.collision_mesh_path).copy()
        coll_mesh.apply_scale(target_object.scales)  # NOTE apply scale first, then rotation and translation
        tf_mat = pose7_to_tf_mat(np.asarray([*target_object.position, *target_object.quaternion], dtype=np.float64))
        coll_mesh.apply_transform(tf_mat)
        if cfg.enable_fps:
            dense_points = coll_mesh.sample(10 * cfg.num_points)
            sampled_indices = fpsample.bucket_fps_kdline_sampling(dense_points, cfg.num_points, h=3)
            sampled_points = dense_points[np.asarray(sampled_indices, dtype=np.int64)]
        else:
            sampled_points = coll_mesh.sample(cfg.num_points)
        return sampled_points

    if cfg.save_as_separate_files:
        for scene_id, scene_description in track(scene_descriptions.items(), description="Sampling..."):
            output_filepath = os.path.join(cfg.output_path, f"{scene_id}.h5")
            with h5py.File(output_filepath, "w") as f:
                sampled_points = get_pcd(scene_description)
                obs_group = f.create_group("obs")
                obs_group.create_dataset(
                    "pointcloud", data=sampled_points.astype(np.float32), compression="gzip", compression_opts=4
                )
                obs_group.create_dataset(
                    "pointcloud_segmentation",
                    data=np.ones((sampled_points.shape[0], 1), dtype=np.int32),
                    compression="gzip",
                    compression_opts=4,
                )
                obs_group.create_dataset("target_actor_segmentation_id", data=1)
    else:
        with h5py.File(cfg.output_path, "w") as f:
            data_group = f.create_group("data")
            for scene_id, scene_description in track(scene_descriptions.items(), description="Sampling..."):
                sampled_points = get_pcd(scene_description)
                data_group.create_group(scene_id)
                obs_group = data_group[scene_id].create_group("obs")
                obs_group.create_dataset(
                    "pointcloud",
                    data=sampled_points.astype(np.float32)[np.newaxis],
                    compression="gzip",
                    compression_opts=4,
                )
                obs_group.create_dataset(
                    "pointcloud_segmentation",
                    data=np.ones((1, sampled_points.shape[0], 1), dtype=np.int32),
                    compression="gzip",
                    compression_opts=4,
                )
                obs_group.create_dataset("target_actor_segmentation_id", data=1)

    write_completion_marker(cfg.output_path + SAMPLE_POINT_CLOUD_COMPLETE_MARKER)


CONFIGS: Dict[str, Config] = {
    # ----------------------------- eval --------------------------------
    "egad_eval_40_scenes": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/egad_eval/40_scenes.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/egad_eval_40_scenes.h5"),
    ),
    "kit_20_scenes": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/kit/20_scenes.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/kit_20_scenes.h5"),
    ),
    "multidex_200_scenes": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/multidex/200_scenes.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/multidex_200_scenes.h5"),
    ),
    "dexgraspnet_eval_250_scenes": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/dexgraspnet_eval/250_scenes.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dexgraspnet_eval_250_scenes.h5"),
    ),
    "dex1b_test_v1": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/test_v1.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dex1b_test_v1.h5"),
    ),
    # ----------------------------- train --------------------------------
    "dex1b_v1": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v1.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dex1b_v1.h5"),
        num_points=4096,
    ),
    "dex1b_v2": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v2.msgpack"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dex1b_v2"),
        save_as_separate_files=True,
        num_points=2048,
    ),
    "dex1b_floating_mini": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/dex1b/v0.3/floating_mini.msgpack"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/dex1b_floating_mini.h5"),
        save_as_separate_files=False,
        num_points=2048,
    ),
    "ycb_objaverse_7k_scenes": Config(
        scene_description_path=str(DATA_ROOT / "scene_description/ycb_objaverse/7k_scenes.json"),
        output_path=str(PROJECT_ROOT / "data/object_data/full_pointcloud/ycb_objaverse_7k_scenes.h5"),
    ),
}

if __name__ == "__main__":
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS), use_underscores=True)
    sample_point_cloud(cfg)
