import logging
from dataclasses import replace

from robokit.assets.robots.assembly import vega_sharpa as vega_sharpa_assets
from robokit.assets.robots.end_effectors import sharpa_hand as sharpa_hand_assets

from graspkit.assets import calib
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.hand_mount import urdf_with_hand_mount

RIGHT_HAND_MOUNT_PATH = calib.calib_file("vega_sharpa/26-08-30-tl-right_hand_mount.json")
"""Calibrated `R_ee_to_right_hand` mount (URDF ships identity, ~9 deg off the real hand); the left stays nominal."""

# One patched URDF serves both arms: it corrects only the right mount and leaves the left as-is.
if not RIGHT_HAND_MOUNT_PATH.exists():
    # Loud on purpose: a silent fallback trains or deploys a 9-deg-wrong hand still logged as vega_sharpa_right.
    logging.getLogger("graspkit").warning(
        f"hand mount: {RIGHT_HAND_MOUNT_PATH.name} not found, using the NOMINAL identity mount. "
        "The modelled right hand will sit ~9 deg from the real one."
    )
_URDF_PATH, _MESH_DIR = urdf_with_hand_mount(
    vega_sharpa_assets.URDF_PATH, RIGHT_HAND_MOUNT_PATH if RIGHT_HAND_MOUNT_PATH.exists() else None
)

# Rest pose measured on the real robot (26-08-06); the head-camera calibration is only valid here.
INIT_QPOS = (
    (0.0,) * 6  # wheels
    + (0.8997, 1.5697, 0.0993)  # torso
    + (0.2062, -0.0269, 0.0154)  # head
    + (0.8399, 0.5101, 0.3701, -1.3, -0.6501, -0.29, -0.0301)  # L_arm
    + (-0.8399, -0.51, -0.37, -1.3, 0.65, 0.29, 0.0301)  # R_arm
    + (0.0,) * 44  # right_hand, left_hand
)

# prunes chain to R_arm_j1..7 + right hand; reuses sharpa_right spheres/contacts, so no vega-body self-collision
vega_sharpa_right = RobotConfig(
    name="vega_sharpa_right",
    urdf_or_mjcf_path=_URDF_PATH,
    mesh_dir=_MESH_DIR,
    base_type="fixed",
    init_position=(-0.5, 0.0, -0.85),  # 10cm closer to the table, 10cm lower, centered on it
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    # FK shoulder position at rest pose; recompute if init_position or INIT_QPOS torso/arm joints change
    workspace_origin=(-0.621, -0.169, 0.495),
    arm_base_link_name="arm_center",
    hand_base_link_name="right_hand_C_MC",
    ee_link_names=(
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ),
    ee_coord_spec=sharpa_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=sharpa_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(vega_sharpa_assets.COLLISION_SPHERE_PATH),
    self_collision_ignore_path=str(vega_sharpa_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    contact_points_path=str(sharpa_hand_assets.CONTACT_POINTS_DIR_RIGHT / "contact_points.json"),
    thumb_link_names=["right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip", "right_thumb_elastomer"],
    middle_link_names=[
        "right_middle_PP",
        "right_middle_MP",
        "right_middle_DP",
        "right_middle_fingertip",
        "right_middle_elastomer",
    ],
    palm_link_name="right_hand_C_MC",
    fingertip_link_names=[
        "right_thumb_elastomer",
        "right_index_elastomer",
        "right_middle_elastomer",
        "right_ring_elastomer",
        "right_pinky_elastomer",
    ],
    init_qpos=INIT_QPOS,
)


# left-hand counterpart, pruned to L_arm_j1..7 + left hand; contact points are hand-authored, not mirrored
vega_sharpa_left = replace(
    vega_sharpa_right,
    name="vega_sharpa_left",
    workspace_origin=(-0.621, 0.169, 0.495),  # L_arm_l1 (the shoulder) at the rest pose
    hand_base_link_name="left_hand_C_MC",
    ee_coord_spec=sharpa_hand_assets.EE_COORD_SPEC_LEFT,
    ee_coord_link_name=sharpa_hand_assets.MOUNT_LINK_NAME_LEFT,
    palm_link_name="left_hand_C_MC",
    ee_link_names=tuple(n.replace("right_", "left_") for n in vega_sharpa_right.ee_link_names),
    contact_points_path=str(sharpa_hand_assets.CONTACT_POINTS_DIR_LEFT / "contact_points.json"),
    thumb_link_names=[n.replace("right_", "left_") for n in vega_sharpa_right.thumb_link_names],
    middle_link_names=[n.replace("right_", "left_") for n in vega_sharpa_right.middle_link_names],
    fingertip_link_names=[n.replace("right_", "left_") for n in vega_sharpa_right.fingertip_link_names],
)


# Both arms in one chain for arm-vs-arm collisions; the torso stays active: freeze it in `start_from_grasp_dof_indices`.
vega_sharpa_bimanual = replace(
    vega_sharpa_right,
    name="vega_sharpa_bimanual",
    hand_base_link_name=None,
    ee_link_names=vega_sharpa_right.ee_link_names + vega_sharpa_left.ee_link_names,
)
