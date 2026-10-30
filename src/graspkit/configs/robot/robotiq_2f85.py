from robokit.assets.robots.end_effectors import robotiq_2f85 as robotiq_assets

from graspkit.configs.robot_config import RobotConfig

# Robotiq 2F-85 parallel gripper: the two inner fingers are the only surfaces that touch an object.
robotiq_2f85 = RobotConfig(
    name="robotiq_2f85",
    urdf_or_mjcf_path=str(robotiq_assets.URDF_PATH),
    base_type="floating",
    ee_coord_spec=robotiq_assets.EE_COORD_SPEC,
    ee_coord_link_name=robotiq_assets.MOUNT_LINK_NAME,
    collision_spheres_path=str(robotiq_assets.COLLISION_SPHERE_PATH),
    contact_points_path=str(robotiq_assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(robotiq_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    init_qpos=(0.0,),  # one driven joint; the other seven jaw joints mimic it
    palm_link_name="robotiq_85_base_link",
    thumb_link_names=["left_inner_finger"],
    middle_link_names=["right_inner_finger"],
    fingertip_link_names=["left_inner_finger", "right_inner_finger"],
)
