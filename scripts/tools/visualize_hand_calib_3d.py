"""Show cached observed and calibrated hand depth as 3D point clouds. No hardware access.

Usage:
    uv run scripts/tools/visualize_hand_calib_3d.py \
        --result-dir /tmp/graspkit-calib-validation/2026-08-31-link-subset
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser

ZED_CALIB_W, ZED_CALIB_H = 1920, 1080
ZED_FX, ZED_FY, ZED_CX, ZED_CY = 759.984009, 759.984009, 994.403442, 538.051331

logging.getLogger("websockets.server").setLevel(logging.CRITICAL)


@dataclass
class Args:
    result_dir: Path
    variant: str = "all_joint_zero_mount_tl"
    port: int = 8080
    max_depth_residual: float = 0.03


def main(args: Args):
    observed_depth = np.load(args.result_dir / "target_depths.npy")
    rendered_depth = np.load(args.result_dir / f"hand_rendered_depths_{args.variant}.npy")
    rendered_mask = np.load(args.result_dir / f"hand_rendered_masks_{args.variant}.npy") > 0.5
    assert observed_depth.shape == rendered_depth.shape == rendered_mask.shape

    count, height, width = observed_depth.shape
    scale = width / ZED_CALIB_W
    assert abs(height / ZED_CALIB_H - scale) < 1e-6
    fx, fy = ZED_FX * scale, ZED_FY * scale
    cx, cy = (ZED_CX + 0.5) * scale - 0.5, (ZED_CY + 0.5) * scale - 0.5
    vv, uu = np.mgrid[:height, :width]

    def points(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = depth[mask]
        return np.stack(((uu[mask] - cx) / fx * z, (vv[mask] - cy) / fy * z, z), axis=-1).astype(np.float32)

    observed_mask = (
        (observed_depth > 0) & rendered_mask & (np.abs(observed_depth - rendered_depth) <= args.max_depth_residual)
    )
    observed = [points(observed_depth[i], observed_mask[i]) for i in range(count)]
    rendered = [points(rendered_depth[i], rendered_mask[i]) for i in range(count)]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/camera", axes_length=0.05, axes_radius=0.002)
    red = np.tile(np.array([255, 45, 45], dtype=np.uint8), (len(observed[0]), 1))
    cyan = np.tile(np.array([0, 205, 255], dtype=np.uint8), (len(rendered[0]), 1))
    observed_cloud = server.scene.add_point_cloud("/observed_hand", observed[0], red, point_size=0.0025)
    rendered_cloud = server.scene.add_point_cloud("/rendered_hand", rendered[0], cyan, point_size=0.0025)
    capture = server.gui.add_slider("Capture", 0, count - 1, 1, 0)
    show_observed = server.gui.add_checkbox("Observed (red)", initial_value=True)
    show_rendered = server.gui.add_checkbox("Calibrated render (cyan)", initial_value=True)

    def update(_event=None):
        i = int(capture.value)
        observed_cloud.points = observed[i]
        observed_cloud.colors = np.tile(np.array([255, 45, 45], dtype=np.uint8), (len(observed[i]), 1))
        rendered_cloud.points = rendered[i]
        rendered_cloud.colors = np.tile(np.array([0, 205, 255], dtype=np.uint8), (len(rendered[i]), 1))
        observed_cloud.visible = show_observed.value
        rendered_cloud.visible = show_rendered.value

    capture.on_update(update)
    show_observed.on_update(update)
    show_rendered.on_update(update)
    print(
        f"Open http://localhost:{args.port} — showing same-pixel depth pairs within "
        f"{100 * args.max_depth_residual:g} cm"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Args))
