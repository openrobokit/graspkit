from dataclasses import replace

from robokit.assets.robots.end_effectors import sharpa_hand as sharpa_hand_assets

from graspkit.configs.robot_config import RobotConfig

# Runtime-sampling fallback for xarm6_sharpa_right; the hand below uses a hand-authored contact_points.json instead.
SHARPA_CONTACT_CANDIDATES = {
    "right_thumb_PP": 10,
    "right_thumb_DP": 20,
    "right_index_MP": 5,
    "right_index_DP": 10,
    "right_middle_MP": 5,
    "right_middle_DP": 10,
    "right_ring_MP": 5,
    "right_ring_DP": 10,
    "right_pinky_MP": 5,
    "right_pinky_DP": 10,
}

# Coord axes from sharpa_hand/config.yaml (wrist_coord_spec); confirmed by MCP joint origins (+z from right_hand_C_MC).
sharpa_right = RobotConfig(
    name="sharpa_right",
    urdf_or_mjcf_path=str(sharpa_hand_assets.URDF_PATH_RIGHT),
    base_type="floating",
    ee_coord_spec=sharpa_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=sharpa_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(sharpa_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    self_collision_ignore_path=str(sharpa_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
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
    # fingertip-distance term keys off collision spheres; *_fingertip links carry none, so use elastomer pads instead.
    fingertip_link_names=[
        "right_thumb_elastomer",
        "right_index_elastomer",
        "right_middle_elastomer",
        "right_ring_elastomer",
        "right_pinky_elastomer",
    ],
    init_qpos=(0.1,) * 22,
)

sharpa_left = replace(
    sharpa_right,
    name="sharpa_left",
    urdf_or_mjcf_path=str(sharpa_hand_assets.URDF_PATH_LEFT),
    ee_coord_spec=sharpa_hand_assets.EE_COORD_SPEC_LEFT,
    ee_coord_link_name=sharpa_hand_assets.MOUNT_LINK_NAME_LEFT,
    collision_spheres_path=str(sharpa_hand_assets.COLLISION_SPHERE_PATH_LEFT),
    self_collision_ignore_path=str(sharpa_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_LEFT),
    contact_points_path=str(sharpa_hand_assets.CONTACT_POINTS_DIR_LEFT / "contact_points.json"),
    thumb_link_names=[name.replace("right_", "left_") for name in sharpa_right.thumb_link_names],
    middle_link_names=[name.replace("right_", "left_") for name in sharpa_right.middle_link_names],
    palm_link_name="left_hand_C_MC",
    fingertip_link_names=[name.replace("right_", "left_") for name in sharpa_right.fingertip_link_names],
)
