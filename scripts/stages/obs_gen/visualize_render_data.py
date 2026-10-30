"""Visualize obs_gen point clouds in viser with world-frame scene meshes/boxes and camera pose (axes + frustum).

Usage:
uv run scripts/stages/obs_gen/visualize_render_data.py --experiment_name g1_inspire_dex1b_v0_4_dev
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import viser

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.io_utils import build_filesystem
from graspkit.utils.viser_utils import visualize_scene_meshes
from robokit.xform.warp.torch_wrappers import pose7_to_tf_mat, tf_mat_to_pose7
from robokit.xform.utils import coord_conversion

logger = logging.getLogger("graspkit")


def main(cfg: ExperimentConfig):
    rendered_data_path = cfg.rendered_data_path
    merged_file = rendered_data_path.rstrip("/") + ".h5"
    data_path = merged_file if build_filesystem(merged_file).exists(merged_file) else rendered_data_path
    logger.info(f"Loading rendered data from: {data_path}")

    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)
    # Scenes hold a variable number of views (V), so load each lazily instead of stacking (ragged V can't stack).
    if data_path.endswith(".h5"):
        import h5py

        with h5py.File(data_path, "r") as f:
            scene_ids = sorted(f.keys())
    else:
        scene_ids = sorted(p[:-3] for p in os.listdir(data_path) if p.endswith(".h5"))
    assert scene_ids, f"No scenes found in {data_path}"

    with open(cfg.renderer.camera_intrinsic_path, "r") as f:
        intr = json.load(f)["intrinsic"]  # (fx, fy, ppx, ppy, height, width, depth_scale)
    fy, height, width = float(intr[1]), int(intr[4]), int(intr[5])
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    aspect = width / height

    # Stored pose is SAPIEN (X-front); right-multiply by R_sapien_opencv for viser's OpenCV (+Z forward) frustum.
    sapien2opencv_rot = coord_conversion("opencv", "sapien", return_tensors="pt")
    T_sapien_opencv = torch.eye(4)
    T_sapien_opencv[:3, :3] = sapien2opencv_rot

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)

    current_scene_id: str | None = None
    scene_pts: np.ndarray | None = None  # (V, Npts, 3)
    scene_cam_opencv: np.ndarray | None = None  # (V, 7)
    pointcloud_handle: viser.PointCloudHandle | None = None

    scene_slider = server.gui.add_slider(label="Scene", min=0, max=max(1, len(scene_ids) - 1), step=1, initial_value=0)
    view_slider = server.gui.add_slider(label="View", min=0, max=1, step=1, initial_value=0)
    point_size_slider = server.gui.add_slider(
        label="Point size", min=0.0005, max=0.01, step=0.0005, initial_value=0.002
    )

    def load_scene(scene_id: str):
        nonlocal scene_pts, scene_cam_opencv
        d = GraspKitData.load(data_path, scene_ids=[scene_id], packed=False)
        scene_pts = d.pointcloud[0].numpy()  # (V, Npts, 3)
        cam_sapien = d.camera_poses[0]  # (V, 7)
        scene_cam_opencv = tf_mat_to_pose7(pose7_to_tf_mat(cam_sapien) @ T_sapien_opencv).numpy()
        view_slider.max = max(1, scene_pts.shape[0] - 1)  # keep min<max (viser slider can't be degenerate)
        if view_slider.value > scene_pts.shape[0] - 1:
            view_slider.value = 0

    def update_visualization():
        nonlocal current_scene_id, pointcloud_handle
        scene_id = scene_ids[int(scene_slider.value)]
        if scene_id != current_scene_id:
            current_scene_id = scene_id
            load_scene(scene_id)
            visualize_scene_meshes(server, scene_descriptions[scene_id])

        v = min(int(view_slider.value), scene_pts.shape[0] - 1)
        cam_pos = scene_cam_opencv[v, :3]
        cam_wxyz = scene_cam_opencv[v, 3:]
        server.scene.remove_by_name("/camera")
        server.scene.add_frame("/camera/frame", position=cam_pos, wxyz=cam_wxyz, axes_length=0.05, axes_radius=0.002)
        server.scene.add_camera_frustum(
            "/camera/frustum",
            fov=fov_y,
            aspect=aspect,
            scale=0.1,
            color=(0, 200, 255),
            position=cam_pos,
            wxyz=cam_wxyz,
        )

        pts = scene_pts[v]
        yellow = np.tile(np.array([255, 255, 0], dtype=np.uint8), (pts.shape[0], 1))
        server.scene.remove_by_name("/scene/pointcloud")
        pointcloud_handle = server.scene.add_point_cloud(
            "/scene/pointcloud",
            points=pts,
            colors=yellow,
            point_size=float(point_size_slider.value),
            point_shape="rounded",
        )

    @scene_slider.on_update
    def _(_event):
        update_visualization()

    @view_slider.on_update
    def _(_event):
        update_visualization()

    @point_size_slider.on_update
    def _(_event):
        if pointcloud_handle is not None:
            pointcloud_handle.point_size = float(point_size_slider.value)

    update_visualization()
    logger.info(f"Visualizing {len(scene_ids)} scenes. Use Scene + View sliders to navigate.")
    input("Press Enter to exit...")


if __name__ == "__main__":

    @dataclass
    class Args:
        experiment_name: str

    args = tyro.cli(Args, use_underscores=True)
    main(build_experiment_config(args.experiment_name))
