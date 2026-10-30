from dataclasses import replace

from robokit.assets.robots.assembly import g1_inspire as g1_inspire_assets

from graspkit.configs.robot_config import RobotConfig

g1_inspire_right = RobotConfig(
    name="g1_inspire_right",
    urdf_or_mjcf_path=str(g1_inspire_assets.URDF_PATH),
    init_position=(-0.4, 0.0, 0.0),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    workspace_origin=(-0.4, -0.4, 0.3),
    arm_base_link_name="torso_link",
    hand_base_link_name="R_hand_base_link",
    ee_link_names=("R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"),
    collision_spheres_path=str(g1_inspire_assets.COLLISION_SPHERE_PATH),
    self_collision_ignore_path=str(g1_inspire_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    contact_points_path=str(g1_inspire_assets.CONTACT_POINTS_PATH),
    contact_avoid_points_path=str(g1_inspire_assets.CONTACT_AVOID_POINTS_PATH),
    init_qpos=(0.0,) * 41,
    ee_coord_spec=g1_inspire_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=g1_inspire_assets.EE_LINK_NAME_RIGHT,
    thumb_link_names=["R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"],
    middle_link_names=["R_middle_proximal", "R_middle_intermediate"],
    palm_link_name="R_hand_base_link",
    # the URDF's 0.5 rad/s finger limit is a placeholder that stretches plans; from 4 rad/s on it no longer binds
    joint_velocity_limits=dict.fromkeys(
        (
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
        ),
        4.0,
    ),
    fingertip_link_names=[
        "R_thumb_distal",
        "R_index_intermediate",
        "R_middle_intermediate",
        "R_ring_intermediate",
        "R_pinky_intermediate",
    ],
)

g1_inspire_right_side_init = replace(
    g1_inspire_right,
    name="g1_inspire_right_side_init",
    init_qpos=(0.0,) * 28 + (0.043, -0.253, -0.827, -0.216, -1.537 + 1, -0.033, -0.303) + (0.0,) * 6,
)
