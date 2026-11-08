"""RealSense L515: pipeline helpers, the background-process streamer, and object-cloud cleanup.

All three need the ``calibration`` extra. The ``short_range`` preset is a sensor fact, not taste: the
L515 default mis-unwraps phase on objects under 1 m and reads them at background distance; short_range
makes the close solution the only valid one, for grasping at 0.3–1.5 m.
"""

import ctypes
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from typing import Literal, Optional

import numpy as np
import pyrealsense2 as rs


def read_intrinsic(profile: rs.pipeline_profile) -> tuple[np.ndarray, float]:
    """The color-stream 3×3 intrinsic and the depth scale (m/unit) of a running ``rs.pipeline_profile``."""
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intrinsic = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    return intrinsic, depth_scale


def start_l515_pipeline(
    realsense_serial: Optional[str] = None,
    color_w: int = 1280,
    color_h: int = 720,
    depth_w: int = 1024,
    depth_h: int = 768,
    fps: int = 30,
    visual_preset: str = "short_range",
    warmup_frames: int = 10,
) -> tuple[rs.pipeline, rs.pipeline_profile, np.ndarray, float, rs.align]:
    """Open an L515 at the project's standard config; returns ``(pipeline, profile, intrinsic, depth_scale, align)``.

    The caller owns ``pipeline.stop()``. ``warmup_frames`` are drained: the laser settles over ~5 frames."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    if realsense_serial is not None:
        cfg.enable_device(realsense_serial)
    cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
    profile = pipeline.start(cfg)
    intrinsic, depth_scale = read_intrinsic(profile)
    align = rs.align(rs.stream.color)
    preset = "default preset"
    if visual_preset is not None:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            preset_enum = getattr(rs.l500_visual_preset, visual_preset, None)
            if preset_enum is not None:
                depth_sensor.set_option(rs.option.visual_preset, float(preset_enum))
                preset = visual_preset
            else:
                print(f"[l515] WARN unknown visual_preset {visual_preset!r}; default left")
    print(f"[l515] {color_w}x{color_h} fx={intrinsic[0, 0]:.1f} depth_scale={depth_scale:.1e} {preset}")

    for _ in range(warmup_frames):
        pipeline.wait_for_frames()
    return pipeline, profile, intrinsic, depth_scale, align


# ------------------------------------------------------------ background streamer

_MAX_W = 1920
_MAX_H = 1080


@dataclass
class RealsenseConfig:
    """Streamer config; ``None`` resolution = the camera's default (see ``rs-enumerate-devices``)."""

    color_width: int | None = None
    color_height: int | None = None
    depth_width: int | None = None
    depth_height: int | None = None
    fps: int = 30
    serial_number: str | None = None
    align_to: Literal["color", "depth"] = "color"


class Realsense:
    """RealSense RGBD camera whose daemon subprocess captures aligned frames into shared arrays."""

    def __init__(self, config: RealsenseConfig):
        self._config = config
        self._color = mp.Array(ctypes.c_uint8, _MAX_H * _MAX_W * 3)
        self._depth = mp.Array(ctypes.c_uint16, _MAX_H * _MAX_W)
        self._intrinsics = mp.Array(ctypes.c_double, 7)
        self._running = mp.Value(ctypes.c_bool, True)
        self._ready = mp.Value(ctypes.c_bool, False)
        self._process: mp.Process | None = None

    def start(self):
        self._process = mp.Process(
            target=Realsense._capture_loop,
            args=(self._config, self._color, self._depth, self._intrinsics, self._running, self._ready),
            daemon=True,
        )
        self._process.start()
        while not self._ready.value:
            if not self._process.is_alive():
                raise RuntimeError("RealSense capture process died — check stream config or camera connection")
            time.sleep(0.01)

    def stop(self):
        self._running.value = False
        if self._process:
            self._process.join(timeout=5.0)

    def get_resolution(self) -> tuple[int, int]:
        """Return active ``(height, width)`` after alignment."""
        return int(self._intrinsics[4]), int(self._intrinsics[5])

    def get_color(self) -> np.ndarray:
        """Return color image as ``(H, W, 3)`` uint8 BGR array."""
        h, w = self.get_resolution()
        return np.array(self._color[: h * w * 3], dtype=np.uint8).reshape(h, w, 3)

    def get_depth(self) -> np.ndarray:
        """Return depth image as ``(H, W)`` uint16 array (raw sensor units)."""
        h, w = self.get_resolution()
        return np.array(self._depth[: h * w], dtype=np.uint16).reshape(h, w)

    def get_intrinsics(self) -> np.ndarray:
        """Return ``[fx, fy, ppx, ppy, height, width, depth_scale]`` as float64."""
        return np.array(self._intrinsics[:])

    @staticmethod
    def _capture_loop(
        config: RealsenseConfig,
        color: SynchronizedArray,
        depth: SynchronizedArray,
        intrinsics: SynchronizedArray,
        running: Synchronized,
        ready: Synchronized,
    ) -> None:

        rs_config = rs.config()
        if config.serial_number is not None:
            rs_config.enable_device(config.serial_number)
        if config.color_width is not None and config.color_height is not None:
            rs_config.enable_stream(
                rs.stream.color, config.color_width, config.color_height, rs.format.bgr8, config.fps
            )
        else:
            rs_config.enable_stream(rs.stream.color, rs.format.bgr8, config.fps)
        if config.depth_width is not None and config.depth_height is not None:
            rs_config.enable_stream(rs.stream.depth, config.depth_width, config.depth_height, rs.format.z16, config.fps)
        else:
            rs_config.enable_stream(rs.stream.depth, rs.format.z16, config.fps)

        pipeline = rs.pipeline()
        profile = pipeline.start(rs_config)

        align = rs.align(rs.stream.color if config.align_to == "color" else rs.stream.depth)

        target_stream = rs.stream.color if config.align_to == "color" else rs.stream.depth
        stream_profile = profile.get_stream(target_stream)
        intr = stream_profile.as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrinsics[:] = [intr.fx, intr.fy, intr.ppx, intr.ppy, intr.height, intr.width, depth_scale]

        n_color = intr.height * intr.width * 3
        n_depth = intr.height * intr.width
        while running.value:
            frameset = align.process(pipeline.wait_for_frames())
            color[:n_color] = np.asarray(frameset.get_color_frame().get_data()).ravel()
            depth[:n_depth] = np.asarray(frameset.get_depth_frame().get_data()).ravel()
            ready.value = True  # after the first frame lands, so a read right after start() is never zeros

        pipeline.stop()


# ------------------------------------------------------------ object-cloud cleanup


def clean_object_pc(
    pc_world: np.ndarray,
    voxel_m: float = 0.005,
    nb: int = 20,
    std: float = 1.0,
    dbscan_eps_m: float = 0.01,
    dbscan_min_points: int = 10,
    min_points: int = 32,
) -> Optional[np.ndarray]:
    """Voxel + statistical-outlier + DBSCAN-largest-cluster cleanup; ``None`` when fewer than ``min_points`` survive.

    Defaults suit L515 single-object clouds at 0.3–1.5 m. No plane RANSAC: SAM already isolated the object."""
    import open3d as o3d

    if pc_world.shape[0] < min_points:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_world.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_m)
    if len(pcd.points) < min_points:
        return None
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    if len(pcd.points) < min_points:
        return None
    # keep only the largest DBSCAN cluster; if every point is noise (label -1) fall back to the post-outlier cloud
    labels = np.asarray(pcd.cluster_dbscan(eps=dbscan_eps_m, min_points=dbscan_min_points))
    valid_labels = labels[labels >= 0]
    if valid_labels.size > 0:
        ids, counts = np.unique(valid_labels, return_counts=True)
        largest = int(ids[counts.argmax()])
        pcd = pcd.select_by_index(np.where(labels == largest)[0].tolist())
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts if pts.shape[0] > 0 else None
