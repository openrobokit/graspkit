from dataclasses import replace

from robokit.assets.robots.assembly import xarm7_ability as xarm7_ability_assets
from robokit.assets.robots.end_effectors import ability_hand

from graspkit.configs.robot.ability import ABILITY_CONTACT_CANDIDATES
from graspkit.configs.robot_config import RobotConfig

xarm7_ability_right = RobotConfig(
    name="xarm7_ability_right",
    urdf_or_mjcf_path=str(xarm7_ability_assets.URDF_GLB_PATH),
    self_collision_ignore_path=str(xarm7_ability_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    workspace_origin=(-0.41, -0.37, 0.01),
    init_position=(-0.41, -0.37, 0.01),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    arm_base_link_name="link_base",
    hand_base_link_name="base",
    ee_coord_spec=ability_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=ability_hand.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(xarm7_ability_assets.COLLISION_SPHERE_PATH),
    contact_candidates_config=ABILITY_CONTACT_CANDIDATES,
    contact_avoid_points_path=str(xarm7_ability_assets.CONTACT_AVOID_POINTS_PATH),
    thumb_link_names=["thumb_L1", "thumb_tip", "thumb_L2"],
    middle_link_names=["middle_L1", "middle_tip", "middle_L2"],
    init_qpos=(-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00, -0.03, 0.0, 0.27, 0.27, 0.27, 0.27),
    palm_link_name="thumb_base",
    # *_tip links carry no collision spheres in this URDF's sphere file; use the *_L2 links instead.
    fingertip_link_names=["thumb_L2", "index_L2", "middle_L2", "ring_L2", "pinky_L2"],
)

xarm7_ability_right_side_init = replace(
    xarm7_ability_right,
    name="xarm7_ability_right_side_init",
    init_qpos=(-0.49, 0.10, -0.43, 0.67, 2.16, 1.26, 0.50, -0.03, 0.01, 0.29, 0.28, 0.28, 0.28),
)

xarm7_ability_right_centered = replace(
    xarm7_ability_right,
    name="xarm7_ability_right_centered",
    init_position=(-0.45, 0.0, 0.0),
    init_qpos=(-0.49, 0.10, -0.43, 0.67, 2.16, 1.26, 0.50, -0.03, 0.01, 0.29, 0.28, 0.28, 0.28),
)
