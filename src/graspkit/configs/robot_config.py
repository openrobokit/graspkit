import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from robokit.utils.ee_coord_utils import GripperCoordinateSpec, HandCoordinateSpec

HandAxis = Literal["palm_forward", "four_fingers_up", "palm_right"]
"""A semantic `hand_coord_spec` axis, named without its `_axis` suffix."""


@dataclass
class RobotConfig:
    """Optimization configuration for a robot."""

    name: str
    """Name of the robot"""

    # --- Paths
    urdf_or_mjcf_path: str
    """Path to the urdf or mjcf file"""
    mesh_dir: Optional[str] = None
    """Path to the mesh directory"""
    collision_spheres_path: Optional[str] = None
    """Path to the collision spheres file"""
    self_collision_ignore_path: Optional[str] = None
    """Ignore YAML; .sphere.yml/.capsule.yml selects the sibling matching the term's representation."""
    contact_points_path: Optional[str] = None
    """Path to the contact points file"""
    contact_link_names: Optional[Tuple[str, ...]] = None
    """Restrict contact candidates to these links (e.g. fingertips only). None: keep every link."""
    contact_avoid_points_path: Optional[str] = None
    """Path to the contact avoid points file"""

    base_type: Literal["floating", "fixed"] = "fixed"
    """Type of the base"""
    keep_world_link: bool = False
    """MJCF only: keep the world body as a fixed base link (for a mjcf root with its own joint, e.g. lift base)."""
    init_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial position of the robot base at reset (x, y, z)"""
    init_quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Initial base-orientation quaternion at reset, real part first (w, x, y, z)"""
    init_qpos: Optional[Tuple[float, ...]] = None
    """Initial qpos for simulation and motion planning"""
    workspace_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Workspace origin (x, y, z), e.g. arm base or right shoulder"""
    joint_velocity_limits: Optional[Dict[str, float]] = None
    """Velocity limits (rad/s) by joint name, overriding the URDF's in motion planning"""

    ee_coord_spec: Optional[Union[HandCoordinateSpec, GripperCoordinateSpec]] = None
    """robokit's end-effector spec as published; `grasp_robot.hand_coord_spec` turns it into the hand frame."""
    ee_coord_link_name: Optional[str] = None
    """The link `ee_coord_spec` is expressed in (robokit's mount link). None: the hand frame itself."""

    # --- Link names
    arm_base_link_name: Optional[str] = None
    """Base link name for the arm, defaults to the robot's base link if not provided"""
    hand_base_link_name: Optional[str] = None
    """Base link name for the hand"""
    ee_link_names: Optional[Tuple[str, ...]] = None
    """Leaf link names the chain must reach; `GraspRobot.from_config` prunes other branches by default."""
    thumb_link_names: Optional[List[str]] = None
    """List of link names for the thumb"""
    middle_link_names: Optional[List[str]] = None
    """List of link names for the middle finger"""
    fingertip_link_names: Optional[List[str]] = None
    """List of link names for the fingertips"""
    palm_link_name: Optional[str] = None
    """Name of the palm link"""

    # --- Contact
    contact_candidates_config: Optional[Dict[str, int]] = None
    """Contact candidates configuration"""
    contact_sampling_qpos: Optional[Tuple[float, ...]] = None
    """Qpos to use when sampling contact points (e.g., for thumb opposition)"""

    def get_self_collision_ignore_path(
        self, representation: Literal["sphere", "capsule"], warn: bool = False
    ) -> Optional[str]:
        path = self.self_collision_ignore_path
        if path is not None:
            Path(path).stat()  # An invalid supplied path is an error, even when selecting its sibling.
            path = path.replace(".sphere.yml", f".{representation}.yml").replace(
                ".capsule.yml", f".{representation}.yml"
            )
            if path != self.self_collision_ignore_path and not Path(path).is_file():
                path = None
        if warn and path is None:
            logging.getLogger("graspkit").warning(
                "Robot '%s': building %s self-collision term without a self-collision ignore path.",
                self.name,
                representation,
            )
        return path

    def __hash__(self) -> int:
        # Hash by name only (list/dict fields block auto-hash); every `replace(...)` variant must set a new name.
        return hash(self.name)
