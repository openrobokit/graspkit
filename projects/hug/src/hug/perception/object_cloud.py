"""Lifting an object point cloud into W0, the both-feet frame the policy works in; no optional dependencies.

Sim (actor-id masks) and real (SAM2) both end with ``pc_w0 = W0_T_cam · pc_cam``. The pinhole unprojection
and both routes to ``W0_T_cam`` — FK to the head camera, or FK to the torso · chest extrinsic — live here.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hug.wbc.pose_estimator import LinkPoseEstimator, motor_q_29_to_32

HEAD_CAMERA_LINK = "camera_optical_frame"
CHEST_MOUNT_LINK = "torso_link"


def unproject_depth_to_cam_xyz(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    max_depth_m: float,
    mask: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Pinhole unprojection of ``depth_m`` (H×W metres) to ``(N, 3) float32`` camera XYZ; ``colors`` only with ``rgb``.

    Drops depth ≤ 0 or ≥ ``max_depth_m`` (L515 drop-outs and background bleed) and pixels outside ``mask``."""
    H, W = depth_m.shape
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_m > 0) & (depth_m < max_depth_m)
    if mask is not None:
        valid = valid & mask.astype(bool)
    z = depth_m[valid].astype(np.float32)
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pc_cam = np.stack([x, y, z], axis=-1)
    colors = rgb[valid] if rgb is not None else None
    return pc_cam, colors


def pc_cam_to_w0(pc_cam: NDArray[np.float64], W0_T_cam: NDArray[np.float64]) -> NDArray[np.float32]:
    """Camera-frame points → W0."""
    return (pc_cam @ W0_T_cam[:3, :3].T + W0_T_cam[:3, 3]).astype(np.float32)


def head_camera_W0_T_cam(
    estimator: LinkPoseEstimator,
    motor_q: NDArray[np.float64],
    head_qpos: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """``W0_T_camera_optical_frame`` by FK; ``head_qpos`` must be the pose the camera was actually rendered from."""
    full_q = motor_q_29_to_32(motor_q) if head_qpos is None else motor_q_29_to_32(motor_q, head_qpos)
    return np.asarray(estimator.estimate(full_q, link_name=HEAD_CAMERA_LINK).matrix, dtype=np.float64)


def chest_camera_W0_T_cam(
    estimator: LinkPoseEstimator,
    motor_q: NDArray[np.float64],
    torso_T_cam: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``W0_T_torso(motor_q) · torso_T_cam`` for the chest-mounted L515 (``torso_T_cam`` from hand-eye calibration)."""
    W0_T_torso = np.asarray(estimator.estimate(motor_q_29_to_32(motor_q), link_name=CHEST_MOUNT_LINK).matrix)
    return W0_T_torso.astype(np.float64) @ torso_T_cam


def load_torso_T_cam(path: str | Path) -> NDArray[np.float64]:
    """Load and check ``T_torso_camera`` from a g1_calibration .json; a bad one would shift every grasp by metres."""
    arr = np.asarray(json.loads(Path(path).read_text())["T_torso_camera"], dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"torso_T_cam at {path} must be 4x4, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"torso_T_cam at {path} contains non-finite values")
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"torso_T_cam at {path} has non-homogeneous bottom row {arr[3]}")
    return arr
