"""Engine-agnostic description of the simulated world: the object catalog, joint layout, reset pose, tunables.

Engine-specific numbers live in :class:`SapienPhysicsConfig`, the task setup in :class:`SceneConfig`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hug import paths

# ---------------------------------------------------------------------- objects


def object_dir(object_name: str) -> Path:
    d = paths.OBJECTS_DIR / object_name
    if not d.is_dir():
        raise FileNotFoundError(f"no object {object_name!r} under {paths.OBJECTS_DIR}")
    return d


def object_mesh_path(object_name: str) -> Path:
    """The merged collision mesh, ``<name>.stl``."""
    stl = object_dir(object_name) / f"{object_name}.stl"
    if not stl.is_file():
        raise FileNotFoundError(f"no mesh for {object_name!r}: expected {stl}")
    return stl


def object_visual_mesh_path(object_name: str) -> Path:
    """Render mesh: ``textured.obj`` when the object ships one, else the collision mesh."""
    textured = object_dir(object_name) / "textured.obj"
    return textured if textured.is_file() else object_mesh_path(object_name)


def object_collision_parts(object_name: str) -> list[Path]:
    """The baked convex decomposition, ``collision/part_*.obj`` in index order."""
    parts = sorted((object_dir(object_name) / "collision").glob("part_*.obj"))
    if not parts:
        raise FileNotFoundError(f"{object_name!r} has no collision/part_*.obj (its baked convex decomposition)")
    return parts


# Per-object uniform scale (absent = true size), sized to the Inspire hand's aperture.
OBJECT_SCALE: dict[str, float] = {
    "honey_bottle": 0.65,
    "master_chef_can": 0.75,
    "pitcher_base": 0.5,
    "plum": 1.6,
    "potted_meat_can": 0.95,
    "rubiks_cube": 1.2,
    "soap_dispenser": 0.75,
    "tomato_soup_can": 0.85,
    "toy_airplane": 0.4,
    "turmeric": 0.65,
    "wood_block": 0.65,
}


def object_scale(object_name: str) -> float:
    return OBJECT_SCALE.get(object_name, 1.0)


def available_objects() -> list[str]:
    """Every object that ships its merged mesh, sorted by name."""
    root = paths.OBJECTS_DIR
    return sorted(d.name for d in root.iterdir() if (d / f"{d.name}.stl").is_file()) if root.is_dir() else []


# ------------------------------------------------------------- robot joint layout

# 29 G1 body joints in unitree_hg G1JointIndex order (legs / waist / arms).
# fmt: off
G1_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",      "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint", "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# fmt: on
N_G1 = 29

# Inspire hand: 6 proximal (commanded) + 6 mimic (derived); order thumb_rotate, thumb_bend, index, middle, ring, pinky.
HAND_PROXIMAL_SUFFIXES: list[str] = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]
HAND_MIMIC_SUFFIXES: list[str] = [
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_intermediate_joint",
    "middle_intermediate_joint",
    "ring_intermediate_joint",
    "pinky_intermediate_joint",
]
HAND_MIMIC_FROM_URDF: list[int] = [1, 1, 2, 3, 4, 5]  # source proximal index per mimic joint
HAND_MIMIC_SCALE: NDArray[np.float64] = np.array([1.334, 0.667, 1.0, 1.0, 1.0, 1.0])

# ------------------------------------------------------------------ reset pose
STAND_BASE_Z = 0.78  # pelvis height with the soles on the floor at the lower policy's default legs
STAND_ARM_QPOS = np.zeros(7, dtype=np.float64)  # both arms straight down
STAND_HAND_QPOS = np.array([1.3, -0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # open: thumb out and back


# ------------------------------------------------------------------ tunables


@dataclass
class SapienPhysicsConfig:
    """PhysX settings of the policy's ManiSkill training env, except object friction (2.0 there)."""

    sim_freq: float = 100.0
    contact_offset: float = 0.005
    solver_position_iterations: int = 30
    solver_velocity_iterations: int = 1

    object_static_friction: float = 0.8
    object_dynamic_friction: float = 0.8
    object_restitution: float = 0.0
    object_mass: float = 0.1

    finger_static_friction: float = 5.0
    finger_dynamic_friction: float = 4.0
    finger_restitution: float = 0.0
    finger_patch_radius: float = 0.1

    hand_drive_stiffness: float = 1e2
    hand_drive_damping: float = 1e1
    hand_drive_force_limit: float = 50.0

    head_drive_stiffness: float = 5e1
    head_drive_damping: float = 5e0
    head_drive_force_limit: float = 25.0

    body_drive_force_limit: float = 1000.0

    joint_armature: float = 0.01

    foot_ground_friction: float = 1.5


@dataclass
class SceneConfig:
    """Table, object placement / randomisation, and what the head camera can see."""

    table_center: tuple[float, float, float] = (0.45, 0.0, 0.75)  # world, box centre
    table_half_size: tuple[float, float, float] = (0.25, 0.25, 0.015)  # ≥ 4 cm margin around every random draw
    object_spawn_yaw: float = 1.57  # rad, about +z
    object_drop_clearance: float = 0.085  # m above the table top at spawn; settles rather than interpenetrating

    random_x_offset: tuple[float, float] = (-0.20, 0.10)  # randomisation, from the table centre
    random_y_offset: tuple[float, float] = (-0.18, 0.12)  # asymmetric: biased toward the grasping (right) hand

    camera_fovy_deg: float = 70.0  # head camera fovy; narrower blinds the near half of a low table
    min_cloud_z_fraction: float = 0.6  # captured object height / true height below which the view counts as clipped
    min_object_pixels: int = 50  # fewer object pixels → "not visible", not a sparse cloud

    @property
    def default_table_top_z(self) -> float:
        return self.table_center[2] + self.table_half_size[2]
