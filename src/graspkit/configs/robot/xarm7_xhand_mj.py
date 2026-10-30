"""Fixed-base xArm7 + xHand on the combined mjcf MjDexGraspEnv evaluates; fully actuated so FK matches MuJoCo."""

from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_assets
from robokit.assets.robots.end_effectors import xhand as xhand_assets

from graspkit.configs.robot_config import RobotConfig

XARM_XHAND_MJ_XML = str(xarm7_xhand_assets.MJ_ASSEMBLY_DIR / "xarm7_xhand.xml")

_ARM_HOME = (-0.03, 0.13, 0.03, 0.24, 3.14, 1.46, 0.00)  # exact copy of urdf xarm7_xhand_right home

xarm7_xhand_right_mj = RobotConfig(
    name="xarm7_xhand_right_mj",
    urdf_or_mjcf_path=XARM_XHAND_MJ_XML,
    self_collision_ignore_path=str(xarm7_xhand_assets.MJ_CAPSULE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    keep_world_link=True,
    workspace_origin=(-0.4, 0.0, 0.12),  # the arm base in the combined mjcf
    arm_base_link_name="link_base",
    hand_base_link_name="right_hand_link",
    ee_link_names=(  # mj model leaves: no *_tip bodies, fingers end at the rotaback links
        "right_hand_thumb_rotaback_link2",
        "right_hand_thumb_rotaback_link1",
        "right_hand_index_rotaback_link2",
        "right_hand_index_rotaback_link1",
        "right_hand_midback_link2",
        "right_hand_midback_link1",
        "right_hand_ringback_link2",
        "right_hand_ringback_link1",
        "right_hand_pinkyback_link2",
        "right_hand_pinkyback_link1",
        "right_hand_back_link",
    ),
    ee_coord_spec=xhand_assets.MJCF_EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=xhand_assets.MJCF_MOUNT_LINK_NAME_RIGHT,
    init_qpos=_ARM_HOME + (0.1,) * 12,
    # Curated per-finger contact points from xarm7_xhand_right, filtered to mjcf link names.
    contact_points_path=str(xhand_assets.MJ_CONTACT_POINTS_PATH_RIGHT),
    contact_avoid_points_path=str(xhand_assets.MJ_CONTACT_AVOID_POINTS_PATH_RIGHT),
    collision_spheres_path=str(xarm7_xhand_assets.COLLISION_SPHERE_PATH),
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
