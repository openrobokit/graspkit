"""Control-stack tuning, split sim / real, and the deployments that compose the stack.

Sim and real share structure, not tuning: paired dataclasses with identical fields, so
a difference is visible. Never add a tuned constant as a module literal on one side only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hug import paths
from hug.grasp.planner import GraspTiming, RealGraspTiming, SimGraspTiming
from hug.sim.scene import STAND_ARM_QPOS, SapienPhysicsConfig, SceneConfig

if TYPE_CHECKING:
    from hug.sim.backend import SapienBackend
    from hug.wbc.controller import WholeBodyController

# ---- shared values. Arm PD, shoulder→wrist, left then right: the real robot's gains, mirrored by sim.
# fmt: off
_ARM_KP_BASE = np.array([80., 80., 40., 60., 40., 40., 40.,
                         80., 80., 40., 60., 40., 40., 40.])
_ARM_KD_BASE = np.array([2., 2., 1., 1., 1., 1., 1.,
                         2., 2., 1., 1., 1., 1., 1.])
# fmt: on
ARM_KP: NDArray[np.float64] = _ARM_KP_BASE * 2.0
ARM_KD: NDArray[np.float64] = _ARM_KD_BASE * 2.5

GRAVITY_COMP_CLIP_NM = 10.0  # symmetric clip on the gravity-comp feed-forward, N·m
WAIST_GRAVITY_CLIP_NM = 30.0  # waist feed-forward clip, N·m; the roll/pitch motors are rated 50
HZ = 50.0  # control rate
N_HAND = 12  # 6 left + 6 right proximal hand joints
DEFAULT_TORSO_HEIGHT = 0.75  # m, pelvis above the feet

# Postures. Arms are 7 joints shoulder→wrist; hands are [thumb_yaw, thumb_pitch, index, middle, ring, pinky].
IDLE_HAND = np.array([1.3, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # real holds: thumb out, fingers relaxed
READY_ARM_QPOS = np.array([0.5, -0.3, -0.6, -0.5, -0.41, 0.0, 0.0], dtype=np.float32)  # right arm, grasp READY
PLACE_ARM_QPOS = np.array([0.5, -0.3, -1.15, -0.3, -0.6, 0.0, 0.0], dtype=np.float32)  # right arm, release


def q21_posture(height: float, left_arm: ArrayLike, right_arm: ArrayLike) -> np.ndarray:
    """IK-layout posture ``[torso(4), head(3), left_arm(7), right_arm(7)]`` with zero waist and head."""
    q21 = np.zeros(21, dtype=np.float32)
    q21[0] = height
    q21[7:14] = left_arm
    q21[14:21] = right_arm
    return q21


# ------------------------------------------------------------------ tuned configs


@dataclass
class PelvisCompensatorConfig:
    """Bounds on the pelvis-mismatch correction; the clamps bind, the rate limits are open at 50 Hz."""

    max_trans_rate: float = 0.2  # m / step
    max_rot_rate: float = 0.5  # rad / step
    max_translation: float = 0.12  # m; a safety net against a bad pelvis estimate: sim widens it, real must not
    max_rotation: float = 0.5  # rad


@dataclass
class ControlConfig:
    """Shared shape; defaults are the values sim and real agree on."""

    hz: float = HZ
    arm_vel_limit_rad_per_step: Optional[float] = None  # max arm-joint step per tick; None = unlimited
    pelvis: PelvisCompensatorConfig = field(default_factory=PelvisCompensatorConfig)
    whole_body_gravity: bool = False  # gravity comp also holds the waist, on the measured pelvis tilt


@dataclass
class SimControlConfig(ControlConfig):
    """Tuned against :class:`~hug.sim.backend.SapienBackend`."""

    # a 0.45 m table drives the mismatch to ~0.18 m; 0.12 would turn the overflow into hand error
    pelvis: PelvisCompensatorConfig = field(default_factory=lambda: PelvisCompensatorConfig(max_translation=0.30))
    whole_body_gravity: bool = True  # the waist sags ~2.8 deg under the arms without it


@dataclass
class RealControlConfig(ControlConfig):
    """Tuned against the G1 + Inspire hands."""

    arm_vel_limit_rad_per_step: Optional[float] = 0.15  # bounds a single-tick arm jump
    pelvis: PelvisCompensatorConfig = field(default_factory=PelvisCompensatorConfig)  # the conservative 0.12, not sim's
    whole_body_gravity: bool = True  # a forward reach loads the waist ~10 N·m; fades in and out at the ramps


# ------------------------------------------------------------------ deployments


@dataclass
class Deployment:
    """Composition root: tuned control config + grasp timing + what the IK tracks."""

    control: ControlConfig = field(default_factory=ControlConfig)
    timing: GraspTiming = field(default_factory=GraspTiming)
    track_hands: str = "right"  # "right" for grasping, "both" for teleop

    def controller(self) -> "WholeBodyController":
        """A loaded :class:`WholeBodyController` wired to this deployment."""
        from hug.wbc.controller import WholeBodyController, WholeBodyControllerConfig  # avoids an import cycle
        from hug.wbc.robokit_ik import RobokitWBCConfig

        wbc = WholeBodyController(
            WholeBodyControllerConfig(
                ik=RobokitWBCConfig(track_hands=self.track_hands),
                pelvis=self.control.pelvis,
                hz=self.control.hz,
                whole_body_gravity=self.control.whole_body_gravity,
            )
        )
        wbc.load()
        return wbc


@dataclass
class SimDeployment(Deployment):
    """SAPIEN: adds the physics and scene configs and the physics URDF."""

    control: ControlConfig = field(default_factory=SimControlConfig)
    timing: GraspTiming = field(default_factory=SimGraspTiming)
    physics: SapienPhysicsConfig = field(default_factory=SapienPhysicsConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    urdf: Path = field(default_factory=lambda: paths.G1_INSPIRE_HEAD_URDF)

    def backend(self, scene_object: Optional[str] = None) -> "SapienBackend":
        """A :class:`SapienBackend` on this deployment's physics and scene."""
        from hug.sim.backend import SapienBackend

        return SapienBackend(
            self.urdf, self.control.hz, scene_object=scene_object, physics=self.physics, scene=self.scene
        )

    @staticmethod
    def stand_q21() -> np.ndarray:
        """IK seed matching the backend's reset pose (arms down), so tick 0 is not a step."""
        return q21_posture(DEFAULT_TORSO_HEIGHT, STAND_ARM_QPOS, STAND_ARM_QPOS)


@dataclass
class RealDeployment(Deployment):
    """G1 + Inspire hands: no physics or scene config."""

    control: ControlConfig = field(default_factory=RealControlConfig)
    timing: GraspTiming = field(default_factory=RealGraspTiming)
