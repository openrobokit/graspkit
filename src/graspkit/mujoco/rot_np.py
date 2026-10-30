"""Numpy pose helpers. PORTED FROM DexGraspBench src/util/rot_util.py (commit 56f4fd0-era)."""

import numpy as np
from robokit.xform.numpy import (
    axis_angle_to_quaternion,
    normalize_quaternion,
    quaternion_invert,
    quaternion_raw_multiply,
    quaternion_to_axis_angle,
    standardize_quaternion,
)


def interpolate_pose(pose1: np.ndarray, pose2: np.ndarray, step: int) -> np.ndarray:
    """Interpolate `[x, y, z, qw, qx, qy, qz]` poses: linear translation, shortest-arc slerp; excludes `pose1`."""
    q1 = normalize_quaternion(pose1[3:7])
    q_rel = standardize_quaternion(quaternion_raw_multiply(quaternion_invert(q1), normalize_quaternion(pose2[3:7])))
    rotvec = np.linspace(0, 1, step + 1)[1:, None] * quaternion_to_axis_angle(q_rel)
    quat_interp = quaternion_raw_multiply(q1, axis_angle_to_quaternion(rotvec))
    return np.concatenate([np.linspace(pose1[:3], pose2[:3], step + 1)[1:], quat_interp], axis=1)


def interpolate_qpos(qpos1: np.ndarray, qpos2: np.ndarray, step: int) -> np.ndarray:
    return np.linspace(qpos1, qpos2, step + 1)[1:]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def compute_delta_qpos(qpos1: np.ndarray, qpos2: np.ndarray) -> tuple:
    """Translation distance and rotation angle (degrees) between `[x, y, z, qw, qx, qy, qz]` poses."""
    delta_pos = np.linalg.norm(qpos1[:3] - qpos2[:3])
    q_rel = quaternion_raw_multiply(qpos2[3:], quaternion_invert(qpos1[3:]))
    return delta_pos, np.degrees(2 * np.arccos(min(abs(q_rel[0]), 1.0)))
