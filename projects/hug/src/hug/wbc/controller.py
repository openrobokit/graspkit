"""Whole-body controller: upper-body IK plus pelvis compensation, in the task frame W0.

Targets are pre-warped by the PelvisCompensator and the torso is solved by the IK.
Compensator: ``delta = clamp(rate_limit(W0_T_pelvis_est * ik_T_pelvis^-1))``, applied as ``delta^-1 * target``.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pinocchio as pin

from hug.control.config import (
    DEFAULT_TORSO_HEIGHT,
    GRAVITY_COMP_CLIP_NM,
    HZ,
    READY_ARM_QPOS,
    WAIST_GRAVITY_CLIP_NM,
    PelvisCompensatorConfig,
    q21_posture,
)
from hug.utils import SE3RateLimiter, clamp_se3, mat_to_pose7, pose7_to_mat
from hug.wbc.pose_estimator import LinkPoseEstimator, LinkPoseEstimatorConfig, motor_q_29_to_32
from hug.wbc.robokit_ik import RobokitWBCConfig, RobokitWBController

# IK pinocchio-model layout: q[0] = height, q[1:4] = waist, left arm 7:14, right arm 26:33.
_HAND_FK = {
    "left": (slice(7, 14), "L_hand_base_link"),
    "right": (slice(26, 33), "R_hand_base_link"),
}


class PelvisCompensator:
    """Closed-loop pre-warp of EE targets by the estimated pelvis mismatch."""

    def __init__(
        self,
        estimator: LinkPoseEstimator,
        controller: RobokitWBController,
        config: Optional[PelvisCompensatorConfig] = None,
    ):
        self.estimator = estimator
        self.controller = controller
        self.config = config if config is not None else PelvisCompensatorConfig()
        self._rate_limiter = SE3RateLimiter(
            max_trans_rate=self.config.max_trans_rate,
            max_rot_rate=self.config.max_rot_rate,
        )
        self.W0_T_pelvis = pin.SE3.Identity()  # the latest estimate, for the gravity comp's tilt
        self.weight = 1.0  # how much of the correction to apply; the teleop demo ramps it to show what it fixes

    def reset(self) -> None:
        self._rate_limiter.reset()

    def compute_delta(self, qpos: np.ndarray) -> pin.SE3:
        """Pelvis mismatch, rate-limited and clamped. Stateful: call exactly once per tick."""
        self.W0_T_pelvis = pin.SE3(self.estimator.estimate(motor_q_29_to_32(qpos)).matrix)
        ik_T_pelvis = self.controller.get_pelvis_frame()
        delta = self._rate_limiter.update(self.W0_T_pelvis * ik_T_pelvis.inverse())
        delta = clamp_se3(delta, self.config.max_translation, self.config.max_rotation)
        return delta if self.weight == 1.0 else pin.exp6(self.weight * pin.log6(delta))

    @staticmethod
    def apply_delta(delta: pin.SE3, target: pin.SE3) -> pin.SE3:
        return delta.inverse() * target

    def correct(self, qpos: np.ndarray, target: pin.SE3) -> pin.SE3:
        """``apply_delta(compute_delta(qpos), target)`` for single-target callers."""
        return self.apply_delta(self.compute_delta(qpos), target)


@dataclass
class WholeBodyControllerConfig:
    ik: RobokitWBCConfig = field(default_factory=RobokitWBCConfig)
    pelvis: PelvisCompensatorConfig = field(default_factory=PelvisCompensatorConfig)
    hz: float = HZ
    whole_body_gravity: bool = False  # see ControlConfig.whole_body_gravity


class WholeBodyController:
    """Pelvis compensation → upper-body IK (torso solved) → gravity comp, plus FK and W0 pose queries."""

    def __init__(self, config: Optional[WholeBodyControllerConfig] = None) -> None:
        config = config if config is not None else WholeBodyControllerConfig()
        self.ik = RobokitWBController(config.ik, dt=1.0 / config.hz)
        self.estimator = LinkPoseEstimator()  # W0-frame FK of any link (camera by default)
        pelvis_est = LinkPoseEstimator(LinkPoseEstimatorConfig(target_link="pelvis"))
        self.compensator = PelvisCompensator(pelvis_est, self.ik, config.pelvis)
        self.whole_body_gravity = config.whole_body_gravity  # mutable: the teleop demo toggles it live

    def load(self) -> None:
        """Lazy-init the IK (robokit/warp setup). Call once after construction."""
        self.ik.load_config()

    def home_q21(self) -> np.ndarray:
        """The grasp READY posture: default torso height, IK default left arm, ``READY_ARM_QPOS`` right."""
        return q21_posture(DEFAULT_TORSO_HEIGHT, self.ik.config.default_left_arm, READY_ARM_QPOS)

    def reset(self, q21: Optional[np.ndarray] = None) -> None:
        """Re-seed the IK filters at ``q21`` (default :meth:`home_q21`)."""
        self.ik.reset(self.home_q21() if q21 is None else q21)

    @staticmethod
    def _pose7_to_se3(pose7: Optional[np.ndarray]) -> Optional[pin.SE3]:
        if pose7 is None:
            return None
        mat = pose7_to_mat(pose7.astype(np.float64))
        return pin.SE3(mat[:3, :3], mat[:3, 3])

    def _fk_hand(self, side: str, q: np.ndarray) -> np.ndarray:
        """4x4 pose of the ``side`` hand base in the IK root frame for a full IK-model ``q``."""
        try:
            _, link = _HAND_FK[side]
        except KeyError:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        pin.forwardKinematics(self.ik.pin_model, self.ik.pin_data, q)
        fid = self.ik.pin_model.getFrameId(link)
        pin.updateFramePlacement(self.ik.pin_model, self.ik.pin_data, fid)
        return self.ik.pin_data.oMf[fid].homogeneous

    def fk_hand_pose7(self, side: str, arm_qpos: np.ndarray) -> np.ndarray:
        """pose7 of the ``side`` hand base in the IK root frame at the default torso height. Call after :meth:`load`."""
        full_q = np.zeros(self.ik.pin_model.nq)
        full_q[0] = DEFAULT_TORSO_HEIGHT
        full_q[_HAND_FK[side][0]] = arm_qpos
        return mat_to_pose7(self._fk_hand(side, full_q))

    def gravity_comp(self, q21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clipped gravity feed-forward at posture ``q21``: 14 arm torques, 3 waist torques.

        With ``whole_body_gravity``: at the commanded waist, gravity tilted by the last pelvis estimate, the waist held
        too. Without: the arms only, on an upright pelvis and zero waist."""
        arm_q = np.asarray(q21[7:21], dtype=np.float64)
        if not self.whole_body_gravity:
            tau = self.ik.solve_gravity_compensation(arm_q)
            return np.clip(tau[:14], -GRAVITY_COMP_CLIP_NM, GRAVITY_COMP_CLIP_NM), np.zeros(3)
        gravity = self.compensator.W0_T_pelvis.rotation.T @ np.array([0.0, 0.0, -9.81])
        tau = self.ik.solve_gravity_compensation(arm_q, tuple(q21[1:4]), tuple(gravity))
        arm = np.clip(tau[:14], -GRAVITY_COMP_CLIP_NM, GRAVITY_COMP_CLIP_NM)
        return arm, np.clip(tau[14:], -WAIST_GRAVITY_CLIP_NM, WAIST_GRAVITY_CLIP_NM)

    def step(
        self,
        motor_q: np.ndarray,
        right_pose7: Optional[np.ndarray] = None,
        left_pose7: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One tick: returns ``(q21, arm_tau_ff, waist_tau_ff)``. A ``None`` wrist leaves that arm to the IK's mask."""
        delta = self.compensator.compute_delta(motor_q)  # one delta per tick, shared by both arms

        def to_se3(pose7: Optional[np.ndarray]) -> Optional[np.ndarray]:
            se3 = self._pose7_to_se3(pose7)
            return None if se3 is None else self.compensator.apply_delta(delta, se3).homogeneous

        q21 = self.ik.update(right_wrist2world_mat=to_se3(right_pose7), left_wrist2world_mat=to_se3(left_pose7))
        return q21, *self.gravity_comp(q21)

    def pelvis_pose7_W0(self, motor_q: np.ndarray) -> np.ndarray:
        """Pelvis pose7 in W0 from the both-feet estimator."""
        est = self.estimator.estimate(motor_q_29_to_32(motor_q), link_name="pelvis")
        return np.concatenate([est.position, est.orientation_wxyz]).astype(np.float64)

    def hand_pose7_W0(self, side: str, motor_q: np.ndarray) -> np.ndarray:
        """Hand-base pose7 in W0: ``W0_T_pelvis`` (estimator) ∘ ``pelvis_T_hand`` (IK model, either hand)."""
        q = np.zeros(self.ik.pin_model.nq)  # height 0 → IK root == pelvis
        q[1:4] = motor_q[12:15]
        q[7:14] = motor_q[15:22]
        q[26:33] = motor_q[22:29]
        pelvis_T_hand = self._fk_hand(side, q)
        W0_T_pelvis = self.estimator.estimate(motor_q_29_to_32(motor_q), link_name="pelvis").matrix
        return mat_to_pose7(W0_T_pelvis @ pelvis_T_hand)
