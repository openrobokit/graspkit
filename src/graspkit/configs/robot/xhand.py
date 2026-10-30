from robokit.assets.robots.end_effectors import xhand as xhand_assets

from graspkit.configs.robot_config import RobotConfig

xhand_right = RobotConfig(
    name="xhand_right",
    urdf_or_mjcf_path=str(xhand_assets.URDF_GLB_PATH_RIGHT),
    base_type="floating",
    ee_coord_spec=xhand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=xhand_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(xhand_assets.COLLISION_SPHERE_PATH_RIGHT),
    self_collision_ignore_path=str(xhand_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    contact_points_path=str(xhand_assets.CONTACT_POINTS_PATH_RIGHT),
    contact_avoid_points_path=str(xhand_assets.CONTACT_AVOID_POINTS_PATH_RIGHT),
    init_qpos=(0.1,) * 12,
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
