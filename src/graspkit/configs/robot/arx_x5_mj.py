"""ARX X5 arm and gripper for MuJoCo."""

from dataclasses import replace

from robokit.assets.robots.assembly import arx_x5 as arx_x5_assets

from graspkit.configs.env_config import HandSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.hand_mount import mjcf_with_hand_mount

ARX_X5_MJ_XML = mjcf_with_hand_mount(arx_x5_assets.MJCF_PATH, joint_mimics=(("joint8", "joint7"),))

arx_x5_arm_mj = RobotConfig(
    name="arx_x5_arm_mj",
    urdf_or_mjcf_path=ARX_X5_MJ_XML,
    collision_spheres_path=str(arx_x5_assets.MJ_COLLISION_SPHERE_PATH),
    contact_points_path=str(arx_x5_assets.MJ_CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(arx_x5_assets.MJ_SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    keep_world_link=True,
    init_position=(0.0, -0.35, 0.0),
    workspace_origin=(0.0, -0.35, 0.0),
    arm_base_link_name="world",
    hand_base_link_name="link6",
    ee_coord_spec=arx_x5_assets.EE_COORD_SPEC,
    init_qpos=(0.0,) * 6 + (0.044,),
    palm_link_name="link6",
    thumb_link_names=["link7"],
    middle_link_names=["link8"],
    fingertip_link_names=["link7", "link8"],
    contact_link_names=("link7", "link8"),
)

arx_x5_gripper_mj = replace(
    arx_x5_arm_mj,
    name="arx_x5_gripper_mj",
    urdf_or_mjcf_path=mjcf_with_hand_mount(ARX_X5_MJ_XML, floating_root="link6"),
    base_type="floating",
    keep_world_link=False,
    init_position=(0.0, 0.0, 0.0),
    workspace_origin=(0.0, 0.0, 0.0),
    arm_base_link_name=None,
    init_qpos=(0.044,),
)

ARX_X5 = HandSpec(
    xml_path=ARX_X5_MJ_XML,
    mocap=False,
    finger_prefix=["link7", "link8"],
    valid_body_name=["link6", "link7", "link8"],
    tip_body_name=["link7", "link8"],
    lift_arm_body="link6",
    arm_qpos_indices=tuple(range(6)),
)
