from robokit.assets.robots.end_effectors import yam_gripper as yam_assets

from graspkit.configs.robot_config import RobotConfig

# I2RT YAM flexible_4310: one motor drives two coupled linear jaws.
yam_gripper = RobotConfig(
    name="yam_gripper",
    urdf_or_mjcf_path=str(yam_assets.URDF_PATH),
    base_type="floating",
    ee_coord_spec=yam_assets.EE_COORD_SPEC,
    ee_coord_link_name=yam_assets.MOUNT_LINK_NAME,
    collision_spheres_path=str(yam_assets.COLLISION_SPHERE_PATH),
    contact_points_path=str(yam_assets.CONTACT_POINTS_PATH),
    self_collision_ignore_path=str(yam_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    init_qpos=(-0.048,),  # Open jaws; zero closes the contact faces.
    palm_link_name="base",
    thumb_link_names=["linear_module"],
    middle_link_names=["linear_module_2"],
    fingertip_link_names=["linear_module", "linear_module_2"],
)
