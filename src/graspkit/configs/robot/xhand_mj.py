"""Floating xHand on the hand-only mjcf, same model MjDexGraspEnv uses; fully actuated, 12 dof."""

from robokit.assets.robots.end_effectors import xhand as xhand_assets

from graspkit.configs.robot_config import RobotConfig

XHAND_MJ_XML = str(xhand_assets.MJCF_PATH_RIGHT)

xhand_right_mj = RobotConfig(
    name="xhand_right_mj",
    urdf_or_mjcf_path=XHAND_MJ_XML,
    self_collision_ignore_path=str(xhand_assets.MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    base_type="floating",
    keep_world_link=True,  # root the floating base at the mjcf worldbody = MjDexGraspEnv's freejoint frame
    ee_coord_spec=xhand_assets.MJCF_EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=xhand_assets.MJCF_MOUNT_LINK_NAME_RIGHT,
    init_qpos=(0.1,) * 12,
    contact_points_path=str(xhand_assets.MJ_CONTACT_POINTS_PATH_RIGHT),
    contact_avoid_points_path=str(xhand_assets.MJ_CONTACT_AVOID_POINTS_PATH_RIGHT),
    collision_spheres_path=str(xhand_assets.COLLISION_SPHERE_PATH_RIGHT),
    thumb_link_names=["right_hand_thumb_rota_link1", "right_hand_thumb_rota_link2"],
    middle_link_names=["right_hand_mid_link1", "right_hand_mid_link2"],
    palm_link_name="right_hand_link",
    fingertip_link_names=[
        "right_hand_thumb_rota_link2",
        "right_hand_index_rota_link2",
        "right_hand_mid_link2",
        "right_hand_ring_link2",
        "right_hand_pinky_link2",
    ],
)
