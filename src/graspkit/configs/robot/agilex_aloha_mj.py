"""AgileX ALOHA arm rig for MuJoCo."""

from dataclasses import replace

from robokit.assets.robots.assembly import agilex_aloha as agilex_aloha_assets

from graspkit.configs.env_config import HandSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.hand_mount import mjcf_with_hand_mount

AGILEX_ALOHA_MJ_XML = mjcf_with_hand_mount(agilex_aloha_assets.MJCF_PATH, joint_mimics=(("fr_joint8", "fr_joint7"),))

agilex_aloha_arm_mj = RobotConfig(
    name="agilex_aloha_arm_mj",
    urdf_or_mjcf_path=AGILEX_ALOHA_MJ_XML,
    collision_spheres_path=str(agilex_aloha_assets.MJ_COLLISION_SPHERE_PATH),
    contact_points_path=str(agilex_aloha_assets.MJ_CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(agilex_aloha_assets.MJ_SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    keep_world_link=True,
    # Mount the front-right arm's first joint at (0, -0.35, 0).
    init_position=(-0.2315, -0.0437, -0.839),
    workspace_origin=(0.0, -0.35, 0.0),
    arm_base_link_name="world",
    hand_base_link_name="fr_link6",
    ee_coord_spec=agilex_aloha_assets.EE_COORD_SPEC,
    init_qpos=(0.0,) * 24 + (0.04765,) + (0.0,) * 12,
    palm_link_name="fr_link6",
    thumb_link_names=["fr_link7"],
    middle_link_names=["fr_link8"],
    fingertip_link_names=["fr_link7", "fr_link8"],
    contact_link_names=("fr_link7", "fr_link8"),
)

agilex_aloha_gripper_mj = replace(
    agilex_aloha_arm_mj,
    name="agilex_aloha_gripper_mj",
    urdf_or_mjcf_path=mjcf_with_hand_mount(AGILEX_ALOHA_MJ_XML, floating_root="fr_link6"),
    base_type="floating",
    keep_world_link=False,
    init_position=(0.0, 0.0, 0.0),
    workspace_origin=(0.0, 0.0, 0.0),
    arm_base_link_name=None,
    init_qpos=(0.04765,),
)

AGILEX_ALOHA = HandSpec(
    xml_path=AGILEX_ALOHA_MJ_XML,
    mocap=False,
    finger_prefix=["fr_link7", "fr_link8"],
    valid_body_name=["fr_link6", "fr_link7", "fr_link8"],
    tip_body_name=["fr_link7", "fr_link8"],
    lift_arm_body="fr_link6",
    arm_qpos_indices=tuple(range(18, 24)),
)
