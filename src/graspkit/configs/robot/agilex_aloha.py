"""AgileX ALOHA vendor front-right arm and floating gripper for ManiSkill."""

from dataclasses import replace

from robokit.assets.robots.assembly import agilex_aloha as assets

from graspkit.configs.robot_config import RobotConfig

agilex_aloha_gripper = RobotConfig(
    name="agilex_aloha_gripper",
    urdf_or_mjcf_path=str(assets.GRIPPER_URDF_PATH),
    collision_spheres_path=str(assets.GRIPPER_COLLISION_SPHERE_PATH),
    contact_points_path=str(assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="floating",
    ee_coord_spec=assets.EE_COORD_SPEC,
    init_qpos=(0.04765,),
    palm_link_name="fr_link6",
    hand_base_link_name="fr_link6",
    thumb_link_names=["fr_link7"],
    middle_link_names=["fr_link8"],
    fingertip_link_names=["fr_link7", "fr_link8"],
    contact_link_names=("fr_link7", "fr_link8"),
    ee_link_names=("fr_link7", "fr_link8"),
)

agilex_aloha_arm = replace(
    agilex_aloha_gripper,
    name="agilex_aloha_arm",
    urdf_or_mjcf_path=str(assets.URDF_PATH),
    collision_spheres_path=str(assets.COLLISION_SPHERE_PATH),
    base_type="fixed",
    arm_base_link_name="fr_base_link",
    init_position=(0.0, -0.35, 0.0),
    workspace_origin=(0.0, -0.35, 0.0),
    init_qpos=(0.0,) * 6 + agilex_aloha_gripper.init_qpos,
)
