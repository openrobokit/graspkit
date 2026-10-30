from robokit.assets.robots.assembly import xarm7_sharpa as xarm7_sharpa_assets
from robokit.assets.robots.end_effectors import sharpa_hand

from graspkit.configs.robot_config import RobotConfig

XARM7_SHARPA_URDF = str(xarm7_sharpa_assets.URDF_PATH)

SHARPA_FINGERTIPS = (
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
)
SHARPA_HAND_INIT = (0.1,) * 22

# 7 arm + 22 Sharpa joints for FK/IK/rendering only; no collision spheres are published for the planner or filters.
xarm7_sharpa_right = RobotConfig(
    name="xarm7_sharpa_right",
    urdf_or_mjcf_path=XARM7_SHARPA_URDF,
    self_collision_ignore_path=str(xarm7_sharpa_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    init_position=(-0.45, 0.0, 0.0),
    workspace_origin=(-0.45, 0.0, 0.0),
    hand_base_link_name="right_hand_C_MC",
    ee_coord_spec=sharpa_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=sharpa_hand.MOUNT_LINK_NAME_RIGHT,
    ee_link_names=SHARPA_FINGERTIPS,
    palm_link_name="right_hand_C_MC",
    init_qpos=(-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00) + SHARPA_HAND_INIT,
)
