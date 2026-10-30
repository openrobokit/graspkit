from robokit.assets.robots.end_effectors import mano_hand as mano_hand_assets

from graspkit.configs.robot_config import RobotConfig

mano_right = RobotConfig(
    name="mano_right",
    urdf_or_mjcf_path=str(mano_hand_assets.URDF_PATH_RIGHT),
    base_type="floating",
    ee_coord_spec=mano_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=mano_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(mano_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    self_collision_ignore_path=str(mano_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    contact_sampling_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.65, 0.0, 0.0),
    contact_avoid_points_path=str(mano_hand_assets.CONTACT_AVOID_POINTS_PATH_RIGHT),
    thumb_link_names=["link_13", "link_14", "link_15"],
    middle_link_names=["link_4", "link_5", "link_6"],
    fingertip_link_names=["link_15", "link_3", "link_6", "link_12", "link_9"],
    palm_link_name="link_0",
    init_qpos=(0.0,) * 16,
)
