from robokit.assets.robots.assembly import xarm7_mano as xarm7_mano_assets
from robokit.assets.robots.end_effectors import mano_hand

from graspkit.configs.robot_config import RobotConfig

xarm7_mano_right = RobotConfig(
    name="xarm7_mano_right",
    urdf_or_mjcf_path=str(xarm7_mano_assets.URDF_PATH),
    self_collision_ignore_path=str(xarm7_mano_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    workspace_origin=(-0.41, -0.37, 0.01),
    init_position=(-0.41, -0.37, 0.01),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    hand_base_link_name="mano_link_0",
    ee_coord_spec=mano_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="mano_" + mano_hand.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(xarm7_mano_assets.COLLISION_SPHERE_1_2_PATH),
    contact_avoid_points_path=str(xarm7_mano_assets.CONTACT_AVOID_POINTS_PATH),
    thumb_link_names=["mano_link_13", "mano_link_14", "mano_link_15"],
    middle_link_names=["mano_link_4", "mano_link_5", "mano_link_6"],
    init_qpos=(-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00) + (0.0,) * 16,
    palm_link_name="mano_link_0",
    fingertip_link_names=["mano_link_15", "mano_link_3", "mano_link_6", "mano_link_12", "mano_link_9"],
)
