from robokit.assets.robots.end_effectors import barrett_hand as barrett_hand_assets

from graspkit.configs.robot_config import RobotConfig

barrett_right = RobotConfig(
    name="barrett_right",
    urdf_or_mjcf_path=str(barrett_hand_assets.URDF_GLB_PATH),
    base_type="floating",
    collision_spheres_path=str(barrett_hand_assets.COLLISION_SPHERE_PATH),
    contact_points_path=str(barrett_hand_assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(barrett_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    ee_coord_spec=barrett_hand_assets.EE_COORD_SPEC,
    ee_coord_link_name=barrett_hand_assets.MOUNT_LINK_NAME,
    thumb_link_names=["finger_3_med_link", "finger_3_dist_link"],
    middle_link_names=["finger_1_prox_link", "finger_1_med_liink", "finger_1_dist_link"],
    palm_link_name="base_link",
    fingertip_link_names=["finger_3_dist_link", "finger_2_dist_link", "finger_1_dist_link"],
    init_qpos=(0.0,) * 8,
    contact_candidates_config={
        "finger_3_med_link": 20,
        "finger_3_dist_link": 20,
        "finger_1_dist_link": 20,
        "finger_2_dist_link": 20,
    },
)
