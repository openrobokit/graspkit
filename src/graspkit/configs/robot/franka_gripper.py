from robokit.assets.robots.arms import franka_panda as franka_panda_assets

from graspkit.configs.robot_config import RobotConfig

franka_gripper = RobotConfig(
    name="franka_gripper",
    urdf_or_mjcf_path=str(franka_panda_assets.URDF_PATH),
    self_collision_ignore_path=str(franka_panda_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    init_position=(-0.41, -0.37, 0.01),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    hand_base_link_name="panda_hand",
    collision_spheres_path=str(franka_panda_assets.COLLISION_SPHERE_PATH),
    init_qpos=(0.01, -0.63, -0.01, -2.26, -0.01, 1.63, 0.79, 0.04),
    ee_coord_spec=franka_panda_assets.HAND_EE_COORD_SPEC,
    ee_coord_link_name=franka_panda_assets.HAND_LINK_NAME,
    palm_link_name="panda_hand",
)
