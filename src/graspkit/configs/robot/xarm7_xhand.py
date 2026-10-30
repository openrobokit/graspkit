from dataclasses import replace

from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_assets
from robokit.assets.robots.end_effectors import xhand

from graspkit.configs.robot_config import RobotConfig

xarm7_xhand_right = RobotConfig(
    name="xarm7_xhand_right",
    urdf_or_mjcf_path=str(xarm7_xhand_assets.URDF_GLB_PATH),
    base_type="fixed",
    arm_base_link_name="link_base",
    workspace_origin=(-0.41, -0.37, 0.01),
    init_position=(-0.41, -0.37, 0.01),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    hand_base_link_name="right_hand_link",
    ee_coord_spec=xhand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=xhand.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(xarm7_xhand_assets.COLLISION_SPHERE_PATH),
    self_collision_ignore_path=str(xarm7_xhand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    contact_points_path=str(xarm7_xhand_assets.CONTACT_POINTS_PATH),
    contact_avoid_points_path=str(xarm7_xhand_assets.CONTACT_AVOID_POINTS_PATH),
    thumb_link_names=[
        "right_hand_thumb_bend_link",
        "right_hand_thumb_rota_link1",
        "right_hand_thumb_rotaback_link1",
        "right_hand_thumb_rota_link2",
        "right_hand_thumb_rotaback_link2",
        "right_hand_thumb_rota_tip",
    ],
    middle_link_names=[
        "right_hand_mid_link1",
        "right_hand_midback_link1",
        "right_hand_mid_link2",
        "right_hand_midback_link2",
        "right_hand_mid_tip",
    ],
    init_qpos=(-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00) + (0.1,) * 12,
    palm_link_name="right_hand_link",
    # `*_tip` points sit 3.6 mm inside the distal phalanx spheres and never touch; the phalanx links carry tip geometry
    fingertip_link_names=[
        "right_hand_thumb_rota_link2",
        "right_hand_index_rota_link2",
        "right_hand_mid_link2",
        "right_hand_ring_link2",
        "right_hand_pinky_link2",
    ],
)

# Centered like the mjcf xarm7_xhand_right_mj: arm base at (-0.4, 0, 0.12), reaching object head-on (not off-center).
xarm7_xhand_right_centered = replace(
    xarm7_xhand_right,
    name="xarm7_xhand_right_centered",
    workspace_origin=(-0.4, 0.0, 0.12),
    init_position=(-0.4, 0.0, 0.12),
)
