"""Visualize hand-eye calibration: RealSense point cloud + xArm URDF in viser.

uv run --extra xarm scripts/deploy/visualize_calib.py
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro
import viser
import yourdfpy
from retargetingkit.sources.cameras.realsense import Realsense, RealsenseConfig
from retargetingkit.targets.arms.xarm7 import XArm7, XArm7Config
from robokit.assets.robots.arms import xarm7 as xarm7_assets
from robokit.xform.warp.torch_wrappers import transform_points
from viser.extras import ViserUrdf

from graspkit.constants import ASSETS_ROOT, ROBOT_COLOR


@dataclass
class Args:
    camera_pose_path: str = str(ASSETS_ROOT / "calib/xarm/26-03-24-camera_extrinsic.json")
    """Path to camera extrinsic calibration json"""
    camera_intrinsics_path: str = str(ASSETS_ROOT / "calib/xarm/25-02-19-camera_intrinsic.json")
    """Path to camera intrinsic calibration json"""
    urdf_path: str = str(xarm7_assets.URDF_GLB_PATH)
    """Path to robot URDF"""
    init_qpos: tuple[float, ...] = (-0.031479, 0.134515, 0.031416, 0.23911, 3.141593, 1.464327, -0.003491)
    """Initial joint angles (7-DOF xArm)"""
    xarm_ip: str = "192.168.1.242"
    """xArm7 controller IP address"""
    max_depth: float = 1.5
    """Maximum depth in meters to display"""
    max_points: int = 100000
    """Maximum number of points to display"""


if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)

    # Load camera calibration
    with open(args.camera_pose_path) as f:
        cam_pose = np.asarray(json.load(f)["extrinsic"], dtype=np.float32)
    with open(args.camera_intrinsics_path) as f:
        intrinsics = np.asarray(json.load(f)["intrinsic"], dtype=np.float64)

    # Set up viser + URDF
    urdf = yourdfpy.URDF.load(args.urdf_path, build_scene_graph=True, load_meshes=True)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot", mesh_color_override=ROBOT_COLOR)
    urdf_vis.update_cfg(np.asarray(args.init_qpos, dtype=np.float32))

    # Start hardware
    camera = Realsense(RealsenseConfig())
    camera.start()
    arm = XArm7(XArm7Config(ip=args.xarm_ip))
    arm.start()

    # Intrinsics from calibration (may differ from camera stream resolution)
    fx, fy, ppx, ppy, calib_h, calib_w, depth_scale = intrinsics

    cam_pose_torch = torch.from_numpy(cam_pose)
    pc_handle = None
    xs, ys = None, None

    while True:
        depth = camera.get_depth()
        color = camera.get_color()

        # Build pixel grid on first frame (scaled to actual resolution)
        if xs is None:
            actual_h, actual_w = depth.shape[:2]
            sy, sx = actual_h / calib_h, actual_w / calib_w
            fx, fy, ppx, ppy = fx * sx, fy * sy, ppx * sx, ppy * sy
            ys, xs = np.meshgrid(np.arange(actual_h), np.arange(actual_w), indexing="ij")

        # Unproject depth to 3D
        z = depth.astype(np.float32) * depth_scale
        valid = (z > 0) & (z < args.max_depth)
        x = (xs[valid] - ppx) * z[valid] / fx
        y = (ys[valid] - ppy) * z[valid] / fy
        points = np.stack([x, y, z[valid]], axis=-1).astype(np.float32)
        colors = color[valid]  # (N, 3) uint8 BGR
        if len(points) == 0:
            time.sleep(0.1)
            continue

        # Transform to world frame
        pts_world = transform_points(torch.from_numpy(points), cam_pose_torch).numpy()

        # Subsample for display
        if len(pts_world) > args.max_points:
            idx = np.random.choice(len(pts_world), args.max_points, replace=False)
            pts_world = pts_world[idx]
            colors = colors[idx]

        # Update point cloud
        if pc_handle is not None:
            pc_handle.remove()
        pc_handle = server.scene.add_point_cloud("/point_cloud", pts_world, colors[:, ::-1], point_size=0.002)

        # Update robot joint angles
        urdf_vis.update_cfg(arm.get_qpos())

        time.sleep(0.1)
