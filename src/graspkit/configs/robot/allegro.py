from robokit.assets.robots.end_effectors import allegro_hand as allegro_hand_assets

from graspkit.configs.robot_config import RobotConfig

allegro_right = RobotConfig(
    name="allegro_right",
    urdf_or_mjcf_path=str(allegro_hand_assets.URDF_GLB_PATH_RIGHT),
    self_collision_ignore_path=str(allegro_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    base_type="floating",
    init_qpos=(
        0.0,
        0.44,
        0.33,
        0.52,
        0.0,
        0.19,
        0.73,
        0.15,
        0.0,
        0.29,
        0.56,
        0.38,
        1.2,
        0.0,
        0.0,
        0.0,
    ),
    ee_coord_spec=allegro_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=allegro_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(allegro_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    thumb_link_names=["link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"],
    middle_link_names=["link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip"],
    palm_link_name="base_link",
    fingertip_link_names=["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
)
