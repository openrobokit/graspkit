from robokit.assets.robots.end_effectors import inspire_hand as inspire_hand_assets

from graspkit.configs.robot_config import RobotConfig

inspire_right = RobotConfig(
    name="inspire_right",
    urdf_or_mjcf_path=str(inspire_hand_assets.URDF_PATH_RIGHT),
    base_type="floating",
    init_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ee_coord_spec=inspire_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=inspire_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(inspire_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    self_collision_ignore_path=str(inspire_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    contact_points_path=str(inspire_hand_assets.CONTACT_POINTS_PATH_RIGHT),
    contact_avoid_points_path=str(inspire_hand_assets.CONTACT_AVOID_POINTS_PATH_RIGHT),
    thumb_link_names=["thumb_intermediate", "thumb_proximal", "thumb_distal"],
    middle_link_names=["middle_proximal", "middle_intermediate"],
    palm_link_name="hand_base_link",
    fingertip_link_names=[
        "thumb_distal",
        "index_intermediate",
        "middle_intermediate",
        "ring_intermediate",
        "pinky_intermediate",
    ],
    contact_candidates_config={
        "thumb_intermediate": 20,
        "index_intermediate": 20,
        "middle_intermediate": 20,
        "ring_intermediate": 20,
        "pinky_intermediate": 20,
    },
)
