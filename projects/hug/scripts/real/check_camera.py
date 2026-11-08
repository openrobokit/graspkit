"""L515 chest camera check: no robot needed.

``info`` prints the intrinsics (expect 1280x720, fx near 900, depth_scale≈2.5e-4); ``snapshot`` saves one
colour + depth frame to ``results/realsense/``; ``stream`` shows RGB-D live (ESC exits). The stream is the
calibration capture config, so this hits exactly what ``calibration/g1_calibration.py`` will.

Usage:
    uv run --extra real-cu128 python scripts/real/check_camera.py --mode info
"""

import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import tyro

from hug import paths
from hug.perception.realsense import Realsense, RealsenseConfig

CAPTURE_CONFIG = RealsenseConfig(
    color_width=1280,
    color_height=720,
    depth_width=1024,
    depth_height=768,
    fps=30,
    align_to="color",
)


def info() -> None:
    cam = Realsense(CAPTURE_CONFIG)
    cam.start()
    try:
        intr = cam.get_intrinsics()
        fx, fy, ppx, ppy, h, w, depth_scale = intr
        print(f"resolution:  {int(w)} x {int(h)}  (post-align to color)")
        print(f"fx, fy:      {fx:.4f}, {fy:.4f}")
        print(f"ppx, ppy:    {ppx:.4f}, {ppy:.4f}")
        print(f"depth_scale: {depth_scale:.6f} m/unit")
        # L515 reports ~0.00025; D-series reports ~0.001. Sanity-flag.
        if not (1e-5 < depth_scale < 1e-2):
            print(f"  WARN: depth_scale {depth_scale} looks odd")
    finally:
        cam.stop()


def snapshot() -> None:
    cam = Realsense(CAPTURE_CONFIG)
    cam.start()
    try:
        # First few frames after a fresh stream can be black or auto-exposing.
        time.sleep(0.5)
        color = cam.get_color()
        depth = cam.get_depth()
        scale = cam.get_intrinsics()[6]

        out_color = paths.RESULTS / "realsense" / "color.png"
        out_depth = paths.RESULTS / "realsense" / "depth.npy"
        out_color.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_color), color)
        np.save(out_depth, depth)
        print(f"saved {out_color}  shape={color.shape}  dtype={color.dtype}")
        print(f"saved {out_depth}  shape={depth.shape}  dtype={depth.dtype}  scale={scale:.6f}")
        valid = depth > 0
        if valid.any():
            d_m = depth[valid].astype(np.float32) * scale
            print(
                f"depth (m): min={d_m.min():.3f}  median={np.median(d_m):.3f}  "
                f"max={d_m.max():.3f}  valid={valid.mean() * 100:.1f}%"
            )
        else:
            print("WARN: zero valid depth pixels — check L515 IR LiDAR is powered and unobstructed")
    finally:
        cam.stop()


def stream() -> None:
    cam = Realsense(CAPTURE_CONFIG)
    cam.start()
    try:
        h, w = cam.get_resolution()
        cv2.namedWindow("RGBD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGBD", w * 2, h)
        print("streaming — ESC in the window to exit.")
        while True:
            color = cam.get_color()
            depth = cam.get_depth()
            depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            combined = np.hstack((color, depth_color))
            cv2.imshow("RGBD", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


@dataclass
class Args:
    mode: Literal["info", "snapshot", "stream"] = "info"  # what to check


if __name__ == "__main__":
    {"info": info, "snapshot": snapshot, "stream": stream}[tyro.cli(Args).mode]()
