"""Browse saved grasps (disk/S3) in viser, plus every stage in `<grasp dir>.viz/` from `optimize_grasps.py --verbose`.

Usage:
uv run scripts/stages/grasp_optim/visualize_grasps.py --experiment_name arx_x5_unified_depth
uv run scripts/stages/grasp_optim/visualize_grasps.py --experiment_name arx_x5_unified_depth \
    --grasp_path ~/.graspkit/data/runs/retrain_20261015/gt/arx_x5 --scenes 0 1 2
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro

from graspkit.configs.experiment import build_experiment_config
from graspkit.constants import PROJECT_ROOT
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.datasets.replay_data import ReplayData
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_optim.optimize_pipeline import resolve_robot, static_replays
from graspkit.utils.io_utils import build_filesystem
from graspkit.utils.viser_utils import ReplayViewer


def main(
    experiment_name: str,
    robot_base: Optional[Literal["floating", "fixed"]] = None,
    grasp_path: Optional[str] = None,
    scenes: tuple[int, ...] = (0, 1, 2, 3),
    port: int = 8080,
):
    """grasp_path: per-scene grasp dir (default: the experiment's); scenes: indices into its sorted scene ids."""
    cfg = build_experiment_config(experiment_name)
    robot_config, output_dir = resolve_robot(cfg, robot_base)
    grasp_dir = (grasp_path if grasp_path is not None else output_dir).rstrip("/")
    viz_dir = grasp_dir + ".viz"
    fs = build_filesystem(grasp_dir)
    dirs = [d for d in (grasp_dir, viz_dir) if fs.exists(d)]
    scene_ids = sorted({Path(k).stem for d in dirs for k in fs.ls(d, detail=False) if k.endswith(".h5")})
    descs = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    replays = []
    for scene_id in [scene_ids[i] for i in scenes]:
        if fs.exists(f"{viz_dir}/{scene_id}.h5"):  # asset paths point at the writer's checkout; use local ones
            mesh = descs[scene_id].meshes[descs[scene_id].target_actor_id].collision_mesh_path
            paths = {"robot_path": robot_config.urdf_or_mjcf_path, "mesh_dir": robot_config.mesh_dir or ""}
            paths["obj_mesh_path"] = str(Path(PROJECT_ROOT) / mesh)
            replays += [replace(r, **paths) for r in ReplayData.load(f"{viz_dir}/{scene_id}.h5")]
        if fs.exists(f"{grasp_dir}/{scene_id}.h5"):
            data = GraspKitData.load(f"{grasp_dir}/{scene_id}.h5")
            qpos = torch.stack([data.pre_grasp_qpos, data.grasp_qpos, data.post_grasp_qpos], dim=1)
            replays += static_replays(qpos, robot_config, descs[scene_id], "saved", ["success"] * len(qpos))
    print(f"{len(scene_ids)} scenes in {dirs}; showing {len(replays)} replays of scenes {list(scenes)}")

    viewer = ReplayViewer(port=port)
    viewer.set_replays(replays)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main, use_underscores=True)
