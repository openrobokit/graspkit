from robokit.assets.robots.end_effectors import schunk_hand as schunk_hand_assets

from graspkit.configs.robot_config import RobotConfig

schunk_right = RobotConfig(
    name="schunk_right",
    urdf_or_mjcf_path=str(schunk_hand_assets.URDF_GLB_PATH_RIGHT),
    base_type="floating",
    collision_spheres_path=str(schunk_hand_assets.COLLISION_SPHERE_PATH_RIGHT),
    contact_points_path=str(schunk_hand_assets.CONTACT_POINTS_PATH_RIGHT),
    self_collision_ignore_path=str(schunk_hand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    ee_coord_spec=schunk_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=schunk_hand_assets.MOUNT_LINK_NAME_RIGHT,
    thumb_link_names=["right_hand_a", "right_hand_b", "right_hand_c", "thtip"],
    middle_link_names=["right_hand_k", "right_hand_o", "right_hand_s", "mftip"],
    palm_link_name="right_hand_e1",
    # the *tip links are empty frames; the distal phalanges carry the fingertip spheres
    fingertip_link_names=["right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q"],
    init_qpos=(0.0,) * 9,
)
