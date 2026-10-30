"""ARX X5A vendor arm and floating gripper for ManiSkill."""

from dataclasses import replace

from robokit.assets.robots.assembly import arx_x5 as assets

from graspkit.configs.robot_config import RobotConfig

arx_x5_gripper = RobotConfig(
    name="arx_x5_gripper",
    urdf_or_mjcf_path=str(assets.GRIPPER_URDF_PATH),
    collision_spheres_path=str(assets.GRIPPER_COLLISION_SPHERE_PATH),
    contact_points_path=str(assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="floating",
    ee_coord_spec=assets.EE_COORD_SPEC,
    init_qpos=(0.044,),
    palm_link_name="link6",
    hand_base_link_name="link6",
    thumb_link_names=["link7"],
    middle_link_names=["link8"],
    fingertip_link_names=["link7", "link8"],
    contact_link_names=("link7", "link8"),
    ee_link_names=("link7", "link8"),
)

arx_x5_arm = replace(
    arx_x5_gripper,
    name="arx_x5_arm",
    urdf_or_mjcf_path=str(assets.URDF_PATH),
    collision_spheres_path=str(assets.COLLISION_SPHERE_PATH),
    base_type="fixed",
    arm_base_link_name="base_link",
    init_position=(0.0, -0.35, 0.0),
    workspace_origin=(0.0, -0.35, 0.0),
    init_qpos=(0.0,) * 6 + arx_x5_gripper.init_qpos,
)
