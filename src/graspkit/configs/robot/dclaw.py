from robokit.assets.robots.end_effectors import dclaw_gripper as dclaw_assets

from graspkit.configs.robot_config import RobotConfig

# Three identical 3-DoF fingers, no thumb (others oppose f3); axes read off the rest pose: tips at +x, f3 alone at -y.
dclaw_right = RobotConfig(
    name="dclaw_right",
    urdf_or_mjcf_path=str(dclaw_assets.URDF_PATH),
    base_type="floating",
    ee_coord_spec=dclaw_assets.EE_COORD_SPEC,
    ee_coord_link_name=dclaw_assets.MOUNT_LINK_NAME,
    collision_spheres_path=str(dclaw_assets.COLLISION_SPHERE_PATH),
    contact_points_path=str(dclaw_assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(dclaw_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    init_qpos=(0.0,) * 9,
    palm_link_name="base_link",
    thumb_link_names=["link_f3_1", "link_f3_2", "link_f3_3", "link_f3_head"],
    middle_link_names=["link_f2_1", "link_f2_2", "link_f2_3", "link_f2_head"],
    # the `*_head` links are massless frames with no collision spheres; the distal links carry them
    fingertip_link_names=["link_f1_3", "link_f2_3", "link_f3_3"],
)
