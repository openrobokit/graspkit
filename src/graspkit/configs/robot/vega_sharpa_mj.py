from dataclasses import replace

from robokit.assets.robots.assembly import vega_sharpa as vega_sharpa_assets
from robokit.assets.robots.end_effectors import sharpa_hand

from graspkit.configs.robot_config import RobotConfig

# Combined Vega+Sharpa MJCF; shared by optimizer + MjDexGraspEnv, so joint order matches.
VEGA_SHARPA_MJ_XML = str(vega_sharpa_assets.MJ_ASSEMBLY_DIR / "vega_sharpa.xml")

# Curated contact points lifted from sharpa_right urdf; sampling picks poor spots.
VEGA_SHARPA_MJ_CONTACT_POINTS = str(vega_sharpa_assets.MJ_CONTACT_POINTS_DIR / "contact_points.json")

# 63 mujoco dofs (Lift, torso, head3, L_arm7, hands22 x2); all-zero avoids the forearm dipping into the table (mp=0%).
_INIT_QPOS = (0.0,) * 63

vega_sharpa_right_mj = RobotConfig(
    name="vega_sharpa_right_mj",
    urdf_or_mjcf_path=VEGA_SHARPA_MJ_XML,
    base_type="fixed",
    keep_world_link=True,
    # Place the world/lift base so the all-zero right palm sits at the tabletop cluster (~world 0,0,0.12), in reach.
    init_position=(-0.692, 0.129, -0.808),  # +0.1 x vs the FK probe = stand closer to the table
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    workspace_origin=(0.0, 0.0, 0.05),
    arm_base_link_name="torso_flip_link",
    hand_base_link_name="rh_right_hand_C_MC",
    ee_coord_spec=sharpa_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="rh_" + sharpa_hand.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(vega_sharpa_assets.MJ_COLLISION_SPHERE_PATH),
    contact_points_path=VEGA_SHARPA_MJ_CONTACT_POINTS,
    thumb_link_names=["rh_right_thumb_PP", "rh_right_thumb_DP"],
    middle_link_names=["rh_right_middle_PP", "rh_right_middle_MP", "rh_right_middle_DP"],
    palm_link_name="rh_right_hand_C_MC",
    fingertip_link_names=[
        "rh_right_thumb_DP",
        "rh_right_index_DP",
        "rh_right_middle_DP",
        "rh_right_ring_DP",
        "rh_right_pinky_DP",
    ],
    init_qpos=_INIT_QPOS,
)


# Left-hand counterpart: mjcf/spheres pruned to L_arm+left hand; y mirrored for the all-zero LEFT palm at the cluster.
vega_sharpa_left_mj = replace(
    vega_sharpa_right_mj,
    name="vega_sharpa_left_mj",
    init_position=(-0.692, -0.129, -0.808),
    hand_base_link_name="lh_left_hand_C_MC",
    ee_coord_spec=sharpa_hand.EE_COORD_SPEC_LEFT,
    ee_coord_link_name="lh_" + sharpa_hand.MOUNT_LINK_NAME_LEFT,
    palm_link_name="lh_left_hand_C_MC",
    contact_points_path=str(vega_sharpa_assets.MJ_CONTACT_POINTS_LEFT_DIR / "contact_points.json"),
    thumb_link_names=[n.replace("rh_right_", "lh_left_") for n in vega_sharpa_right_mj.thumb_link_names],
    middle_link_names=[n.replace("rh_right_", "lh_left_") for n in vega_sharpa_right_mj.middle_link_names],
    fingertip_link_names=[n.replace("rh_right_", "lh_left_") for n in vega_sharpa_right_mj.fingertip_link_names],
)
