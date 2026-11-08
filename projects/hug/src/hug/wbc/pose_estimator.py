"""FK estimator for any link in the both-feet world frame W0, and the 29→32 motor splice it needs.

W0(t) is rebuilt from both feet at every call: ``W0_T_link = o_T_W0^-1 * o_T_link``, o = pinocchio's universe.
Lifted from humanoid_locomanip/hal/pose_estimator/pose_estimator.py, trimmed to the single ``29dof_head``
profile on ``g1_29dof_head.urdf`` (32 joints, right hand only), which holds the camera, pelvis, torso and hand.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from hug import paths

# 32-joint order of g1_29dof_head.urdf: legs (12) + waist (3) + head (3) + arms (14).
# fmt: off
G1_29DOF_HEAD_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",      "waist_pitch_joint",
    "head_yaw_joint",         "head_roll_joint",       "head_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint", "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# fmt: on


@dataclass
class LinkPoseEstimatorConfig:
    urdf_path: Path = field(default_factory=lambda: paths.G1_29DOF_HEAD_URDF)
    target_link: str = "camera_optical_frame"
    left_foot_frame: str = "left_ankle_roll_link"
    right_foot_frame: str = "right_ankle_roll_link"
    left_foot_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    right_foot_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    world_frame_mode: str = "midfeet"  # midfeet (full slerp) | midfeet_yaw (yaw only) | left_foot | right_foot
    joint_name_order: list[str] = field(default_factory=lambda: G1_29DOF_HEAD_JOINT_NAMES.copy())
    height_offset: float = 0.02


@dataclass
class LinkPoseEstimate:
    position: np.ndarray  # (3,) in W0
    orientation_wxyz: np.ndarray  # (4,) in W0
    matrix: np.ndarray  # 4x4 W0_T_link
    link_name: str


class LinkPoseEstimator:
    """Any link's pose in the both-feet frame W0: ``link_name`` per call, default ``camera_optical_frame``."""

    def __init__(self, config: Optional[LinkPoseEstimatorConfig] = None):
        self.config = config if config is not None else LinkPoseEstimatorConfig()

        self.model = pin.buildModelFromUrdf(str(self.config.urdf_path))
        self.data = self.model.createData()

        self._left_frame_id = self.model.getFrameId(self.config.left_foot_frame)
        self._right_frame_id = self.model.getFrameId(self.config.right_foot_frame)
        self._default_target_id = self.model.getFrameId(self.config.target_link)

        self._left_offset = pin.SE3(np.eye(3), np.asarray(self.config.left_foot_offset, dtype=float).reshape(3))
        self._right_offset = pin.SE3(np.eye(3), np.asarray(self.config.right_foot_offset, dtype=float).reshape(3))

        self._model_qpos_joint_names = self._get_model_qpos_joint_names()
        self._qpos_indices = self._build_qpos_indices(self.config.joint_name_order)

    def _get_model_qpos_joint_names(self) -> list[str]:
        names = []
        for joint_id in range(1, self.model.njoints):
            if self.model.joints[joint_id].nq > 0:
                names.append(self.model.names[joint_id])
        return names

    def _build_qpos_indices(self, input_joint_order: list[str]) -> np.ndarray:
        expected = set(self._model_qpos_joint_names)
        provided = set(input_joint_order)
        if expected != provided:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise RuntimeError(f"Joint order mismatch. Missing in input: {missing}. Extra in input: {extra}.")
        return np.array([input_joint_order.index(name) for name in self._model_qpos_joint_names], dtype=int)

    def _to_model_qpos(self, qpos: Iterable[float]) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.shape[0] != len(self._qpos_indices):
            raise ValueError(f"Expected qpos length {len(self._qpos_indices)}; got {qpos.shape[0]}.")
        return qpos[self._qpos_indices]

    @staticmethod
    def _average_yaw(yaw_left: float, yaw_right: float) -> float:
        return float(np.arctan2(np.sin(yaw_left) + np.sin(yaw_right), np.cos(yaw_left) + np.cos(yaw_right)))

    def _make_world_from_feet(self, o_T_left: pin.SE3, o_T_right: pin.SE3) -> pin.SE3:
        mode = self.config.world_frame_mode

        if mode == "left_foot":
            return pin.SE3(o_T_left.rotation, o_T_left.translation)
        if mode == "right_foot":
            return pin.SE3(o_T_right.rotation, o_T_right.translation)

        mid_pos = 0.5 * (o_T_left.translation + o_T_right.translation)

        if mode == "midfeet":
            left_q = pin.Quaternion(o_T_left.rotation)
            right_q = pin.Quaternion(o_T_right.rotation)
            avg_q = left_q.slerp(0.5, right_q)
            return pin.SE3(avg_q.toRotationMatrix(), mid_pos)

        if mode == "midfeet_yaw":
            yaw_left = pin.rpy.matrixToRpy(o_T_left.rotation)[2]
            yaw_right = pin.rpy.matrixToRpy(o_T_right.rotation)[2]
            yaw = self._average_yaw(yaw_left, yaw_right)
            rot = pin.rpy.rpyToMatrix(0.0, 0.0, yaw)
            return pin.SE3(rot, mid_pos)

        raise ValueError(f"Unknown world_frame_mode '{mode}'.")

    def _run_fk(self, qpos: Iterable[float]) -> None:
        qpos_model = self._to_model_qpos(qpos)
        pin.forwardKinematics(self.model, self.data, qpos_model)
        pin.updateFramePlacements(self.model, self.data)

    def estimate(
        self,
        qpos: Iterable[float],
        link_name: Optional[str] = None,
    ) -> LinkPoseEstimate:
        self._run_fk(qpos)

        if link_name is None:
            target_frame_id = self._default_target_id
            link_name = self.config.target_link
        else:
            # getFrameId returns nframes for an unknown name (bare index error); the URDF has no L_hand_base_link.
            if not self.model.existFrame(link_name):
                raise ValueError(f"link {link_name!r} is not a frame in {self.config.urdf_path.name}")
            target_frame_id = self.model.getFrameId(link_name)

        o_T_left = self.data.oMf[self._left_frame_id] * self._left_offset
        o_T_right = self.data.oMf[self._right_frame_id] * self._right_offset
        o_T_link = self.data.oMf[target_frame_id]
        o_T_W0 = self._make_world_from_feet(o_T_left, o_T_right)

        W0_T_link = o_T_W0.inverse() * o_T_link

        position = W0_T_link.translation.copy()
        position[2] += self.config.height_offset
        quat_xyzw = pin.Quaternion(W0_T_link.rotation).coeffs()
        orientation_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)

        matrix = W0_T_link.homogeneous.copy()
        matrix[2, 3] += self.config.height_offset

        return LinkPoseEstimate(
            position=position,
            orientation_wxyz=orientation_wxyz,
            matrix=matrix,
            link_name=link_name,
        )


# ------------------------------ 29-motor (12 legs + 3 waist + 14 arms) → 32-joint splice, head after the waist

N_G1_MOTORS = 29
N_URDF_JOINTS = 32
_HEAD_SLICE = slice(15, 18)

# The head is pinned, never commanded (not in the motor vector); 0.6 rad ≈ 34° down aims the camera at the table.
HEAD_QPOS: NDArray[np.float64] = np.array([0.0, 0.0, 0.6], dtype=np.float64)


def motor_q_29_to_32(
    motor_q: NDArray[np.float64],
    head_qpos: NDArray[np.float64] = HEAD_QPOS,
) -> NDArray[np.float64]:
    """Widen a 29-DoF motor vector to the 32-joint FK URDF order; a 32-DoF input passes through.

    The :data:`HEAD_QPOS` default is right for every caller: only head-chain FK (the camera) depends on it."""
    motor_q = np.asarray(motor_q, dtype=np.float64)
    if motor_q.shape[0] == N_URDF_JOINTS:
        return motor_q
    if motor_q.shape[0] != N_G1_MOTORS:
        raise ValueError(f"expected {N_G1_MOTORS}- or {N_URDF_JOINTS}-DoF qpos, got {motor_q.shape[0]}")
    out = np.zeros(N_URDF_JOINTS, dtype=np.float64)
    out[:15] = motor_q[:15]  # legs(12) + waist(3)
    out[_HEAD_SLICE] = head_qpos
    out[18:] = motor_q[15:]  # arms(14)
    return out
