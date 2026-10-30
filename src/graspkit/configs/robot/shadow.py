from robokit.assets.robots.end_effectors import shadow_hand as shadow_hand_assets

from graspkit.configs.robot_config import RobotConfig

shadow_right = RobotConfig(
    name="shadow_right",
    urdf_or_mjcf_path=str(shadow_hand_assets.MJCF_PATH_RIGHT),
    contact_points_path=str(shadow_hand_assets.MJCF_CONTACT_POINTS_PATH_RIGHT),
    collision_spheres_path=str(shadow_hand_assets.MJCF_COLLISION_SPHERE_PATH_RIGHT),
    ee_coord_spec=shadow_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=shadow_hand_assets.MOUNT_LINK_NAME_RIGHT,
    base_type="floating",
    thumb_link_names=["robot0:thproximal", "robot0:thhub", "robot0:thmiddle", "robot0:thdistal"],
    middle_link_names=["robot0:mfknuckle", "robot0:mfproximal", "robot0:mfmiddle", "robot0:mfdistal"],
    fingertip_link_names=[
        "robot0:thdistal",
        "robot0:ffdistal",
        "robot0:mfdistal",
        "robot0:rfdistal",
        "robot0:lfdistal",
    ],
    palm_link_name="robot0:palm",
    init_qpos=(0.0,) * 22,
)
