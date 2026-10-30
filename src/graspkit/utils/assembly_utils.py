"""Robot configs for any base (arm or humanoid) with any ee (hand or gripper), assembled by robokit."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from robokit.assets.robots.arms import yam
from robokit.assets.robots.assembly import composed

from graspkit.configs.robot import build_robot_config
from graspkit.configs.robot_config import RobotConfig

EE_PREFIX = composed.EE_PREFIX


@dataclass(frozen=True)
class Placement:
    """Where a base stands in the scene and the rest pose it holds there."""

    arm_base_link: str
    init_qpos: tuple[float, ...]
    workspace_origin: tuple[float, float, float]
    init_position: tuple[float, float, float]


PLACEMENTS: dict[str, Placement] = {
    "yam": Placement("base_link", tuple(yam.REST_Q[f"joint{i}"] for i in range(6, 0, -1)), (-0.45, 0.0, 0.0), (-0.45, 0.0, 0.0)),
    "xarm7": Placement("link_base", (-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.0), (-0.41, -0.37, 0.01), (-0.41, -0.37, 0.01)),
    "xarm6": Placement("link_base", (0.0, -0.3, -1.0, 0.0, 1.3, 0.0), (-0.41, -0.37, 0.01), (-0.41, -0.37, 0.01)),
    "franka_panda": Placement("panda_link0", (0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785), (-0.45, 0.0, 0.0), (-0.45, 0.0, 0.0)),
    "ur5e": Placement("base_link", (0.0, -1.2, 1.4, -1.75, -1.57, 0.0), (-0.45, 0.0, 0.0), (-0.45, 0.0, 0.0)),
    "g1": Placement("torso_link", (0.0,) * 22 + (0.3, -0.2, 0.0, 1.5, 0.0, 0.0, 0.0), (-0.4, -0.4, 0.3), (-0.4, 0.0, 0.0)),
    "vega_1": Placement(
        "arm_center",
        (0.0,) * 6 + (0.8997, 1.5697, 0.0993) + (0.0,) * 10 + (-0.8399, -0.51, -0.37, -1.5, 0.65, 0.29, 0.0301),
        (-0.621, -0.169, 0.495), (-0.5, 0.0, -0.85),
    ),
}  # fmt: skip
"""Rest poses keep every hand off a 0.5 m tabletop: G1's right elbow bends up, Vega's past `vega_sharpa_right`'s."""

COMPOSED: dict[str, tuple[str, str]] = {f"{base}_{ee}": (base, ee) for base in PLACEMENTS for ee in composed.EES}
"""Assembled robot name -> (base, ee) over the full cross product; nothing gates a pairing, only the mount changes."""


def assembled_robot_config(name: str) -> RobotConfig:
    """`COMPOSED` names are always assembled, even over a same-named hand-written module; others load their module."""
    if name not in COMPOSED:
        return build_robot_config(name)
    base, ee = COMPOSED[name]
    placement, hand = PLACEMENTS[base], build_robot_config(ee)
    urdf, spheres, root = composed.assemble(base, ee)

    def rename(names):
        return [EE_PREFIX + n for n in names] if names else names

    def rename_json(path):  # contact point files are keyed by link name
        if path is None:
            return None
        out = urdf.parent / Path(path).name
        out.write_text(json.dumps({EE_PREFIX + k: v for k, v in json.loads(Path(path).read_text()).items()}))
        return str(out)

    return replace(
        hand,
        name=f"{name}_assembled",  # its sim agent uid; five pairs share `name` with a hand-written agent
        urdf_or_mjcf_path=str(urdf),
        base_type="fixed",
        arm_base_link_name=placement.arm_base_link,
        workspace_origin=placement.workspace_origin,
        init_position=placement.init_position,
        init_qpos=placement.init_qpos + hand.init_qpos,
        hand_base_link_name=EE_PREFIX + root,
        ee_coord_link_name=EE_PREFIX + hand.ee_coord_link_name if hand.ee_coord_link_name else None,
        palm_link_name=EE_PREFIX + hand.palm_link_name if hand.palm_link_name else None,
        thumb_link_names=rename(hand.thumb_link_names),
        middle_link_names=rename(hand.middle_link_names),
        fingertip_link_names=rename(hand.fingertip_link_names),
        ee_link_names=tuple(rename(list(hand.ee_link_names))) if hand.ee_link_names else None,
        contact_link_names=tuple(rename(list(hand.contact_link_names))) if hand.contact_link_names else None,
        contact_candidates_config=(
            {EE_PREFIX + k: v for k, v in hand.contact_candidates_config.items()}
            if hand.contact_candidates_config
            else None
        ),
        contact_points_path=rename_json(hand.contact_points_path),
        contact_avoid_points_path=rename_json(hand.contact_avoid_points_path),
        collision_spheres_path=None if spheres is None else str(spheres),
        self_collision_ignore_path=None,
    )
