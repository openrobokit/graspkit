from robokit.assets.robots.end_effectors import ability_hand as ability_hand_assets

from graspkit.configs.robot_config import RobotConfig

ABILITY_CONTACT_CANDIDATES = {
    "thumb_L1": 10,
    "thumb_L2": 20,
    "index_L1": 5,
    "index_L2": 10,
    "middle_L1": 5,
    "middle_L2": 10,
    "ring_L1": 5,
    "ring_L2": 10,
    "pinky_L1": 5,
    "pinky_L2": 10,
}

ability_right = RobotConfig(
    name="ability_right",
    urdf_or_mjcf_path=str(ability_hand_assets.URDF_GLB_PATH_RIGHT),
    self_collision_ignore_path=str(ability_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    base_type="floating",
    init_qpos=(-0.03, 0.0, 0.27, 0.27, 0.27, 0.27),
    # 'thumb_q1', 'thumb_q2', 'index_q1', 'middle_q1', 'ring_q1', 'pinky_q1'
    ee_coord_spec=ability_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=ability_hand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(ability_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    contact_candidates_config=ABILITY_CONTACT_CANDIDATES,
    contact_avoid_points_path=str(ability_hand_assets.CONTACT_AVOID_POINTS_PATH_RIGHT),
    thumb_link_names=["thumb_L1", "thumb_L2", "thumb_tip"],
    middle_link_names=["middle_L1", "middle_L2", "middle_tip"],
    palm_link_name="thumb_base",
    # the `*_tip` links are massless frames with no collision spheres; the distal phalanges carry them
    fingertip_link_names=["thumb_L2", "index_L2", "middle_L2", "ring_L2", "pinky_L2"],
)
