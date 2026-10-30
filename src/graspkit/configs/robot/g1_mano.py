from robokit.assets.robots.assembly import g1_mano as g1_mano_assets
from robokit.assets.robots.end_effectors import mano_hand as mano_hand_assets

from graspkit.configs.robot_config import RobotConfig

g1_mano_right = RobotConfig(
    name="g1_mano_right",
    urdf_or_mjcf_path=str(g1_mano_assets.URDF_PATH),
    self_collision_ignore_path=str(g1_mano_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    workspace_origin=(-0.4, -0.2, 0.35),
    init_position=(-0.4, 0.0, 0.05),  # 5cm above the original position
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    arm_base_link_name="torso_link",
    hand_base_link_name="mano_link_0",
    ee_link_names=("mano_link_16_tip", "mano_link_17_tip", "mano_link_18_tip", "mano_link_19_tip", "mano_link_20_tip"),
    ee_coord_spec=mano_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="mano_" + mano_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(g1_mano_assets.COLLISION_SPHERE_PATH),
    contact_points_path=str(g1_mano_assets.CONTACT_POINTS_PATH),
    contact_avoid_points_path=str(g1_mano_assets.CONTACT_AVOID_POINTS_PATH),
    init_qpos=(0.0,) * 45,
    thumb_link_names=["mano_link_13", "mano_link_14", "mano_link_15"],
    middle_link_names=["mano_link_4", "mano_link_5", "mano_link_6"],
    palm_link_name="mano_link_0",
    fingertip_link_names=["mano_link_15", "mano_link_3", "mano_link_6", "mano_link_12", "mano_link_9"],
)
