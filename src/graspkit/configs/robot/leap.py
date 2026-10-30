from robokit.assets.robots.end_effectors import leap_hand as leap_hand_assets

from graspkit.configs.robot_config import RobotConfig

# Contacts span the proximal/middle/distal links and palm; the published fingertip-only points made every grasp a pinch.
LEAP_CONTACTS = {f"{link}{s}": 10 for s in ("", "_2", "_3") for link in ("pip", "dip", "fingertip")} | {
    "thumb_pip": 10,
    "thumb_dip": 10,
    "thumb_fingertip": 16,
    "palm_lower": 20,
}

leap_right = RobotConfig(
    name="leap_right",
    urdf_or_mjcf_path=str(leap_hand_assets.URDF_PATH_RIGHT),
    base_type="floating",
    collision_spheres_path=str(leap_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    contact_candidates_config=LEAP_CONTACTS,
    self_collision_ignore_path=str(leap_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    ee_coord_spec=leap_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=leap_hand_assets.MOUNT_LINK_NAME_RIGHT,
    thumb_link_names=["thumb_pip", "thumb_dip", "thumb_fingertip"],
    middle_link_names=["pip_2", "dip_2", "fingertip_2"],
    palm_link_name="palm_lower",
    fingertip_link_names=["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"],
    init_qpos=(0.0,) * 16,
)
