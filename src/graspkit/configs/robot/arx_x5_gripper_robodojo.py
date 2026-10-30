from robokit.assets.robots.end_effectors import arx_x5_gripper

from graspkit.configs.robot_config import RobotConfig

# RoboDojo's ARX X5 (X5A.urdf) cut at link6 (TCP 0.145 m on +x, wrist-camera mount as collision boxes), jaws open on ±y.
arx_x5_gripper_robodojo = RobotConfig(
    name="arx_x5_gripper_robodojo",
    urdf_or_mjcf_path=str(arx_x5_gripper.URDF_PATH),
    base_type="floating",
    ee_coord_spec=arx_x5_gripper.EE_COORD_SPEC,
    collision_spheres_path=str(arx_x5_gripper.COLLISION_SPHERE_PATH),
    contact_points_path=str(arx_x5_gripper.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(arx_x5_gripper.SPHERE_SELF_COLLISION_IGNORE_PATH),
    init_qpos=(0.044,),  # open; `joint7` is each finger's travel from closed (0), joint8 mimics it
    palm_link_name="link6",
    thumb_link_names=["link7"],
    middle_link_names=["link8"],
    fingertip_link_names=["link7", "link8"],
)
