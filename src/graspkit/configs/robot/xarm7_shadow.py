"""Fixed-base xArm7 + Shadow hand on the SAME mjcf MjDexGraspEnv/MjEvaluator runs, so grasp-opt joints need no remap."""

from robokit.assets.robots.assembly import xarm7_shadow as xarm7_shadow_assets
from robokit.assets.robots.end_effectors import shadow_hand

from graspkit.configs.robot.shadow_mj import SHADOW_MJ_CONTACT_CANDIDATES
from graspkit.configs.robot_config import RobotConfig

XARM_SHADOW_XML = str(xarm7_shadow_assets.MJ_ASSEMBLY_DIR / "xarm7_shadow.xml")

# xArm7 home keyframe + shadow_right curl init (thumb opposition) so grasps start wrapping, matching shadow_right_mj.
_ARM_HOME = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)
_SHADOW_INIT = (
    0.0,
    0.3,
    0.4,
    0.4,  # FF
    0.0,
    0.3,
    0.4,
    0.4,  # MF
    0.0,
    0.3,
    0.4,
    0.4,  # RF
    0.0,
    0.0,
    0.3,
    0.4,
    0.4,  # LF
    0.2,
    0.8,
    0.2,
    0.3,
    0.0,  # TH
)

xarm7_shadow_right = RobotConfig(
    name="xarm7_shadow_right",
    urdf_or_mjcf_path=XARM_SHADOW_XML,
    base_type="fixed",
    # Keeps link_base at its mjcf z=0.12 placement for both optimizer and MjDexGraspEnv (no re-root offset).
    keep_world_link=True,
    # Must match the arm base (link_base at BASE_OFFSET=(-0.5,0,0.12) in the combined mjcf), else grasps miss reach.
    workspace_origin=(-0.35, 0.0, 0.12),
    arm_base_link_name="link_base",
    hand_base_link_name="rh_palm_r",
    ee_coord_spec=shadow_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="rh_palm_r",  # the MJCF palm, oriented like the URDF forearm,
    init_qpos=_ARM_HOME + _SHADOW_INIT,
    collision_spheres_path=str(xarm7_shadow_assets.MJ_COLLISION_SPHERE_PATH),
    contact_candidates_config={f"{k}_r": v for k, v in SHADOW_MJ_CONTACT_CANDIDATES.items()},
    thumb_link_names=["rh_thproximal_r", "rh_thhub_r", "rh_thmiddle_r", "rh_thdistal_r"],
    middle_link_names=["rh_mfknuckle_r", "rh_mfproximal_r", "rh_mfmiddle_r", "rh_mfdistal_r"],
    palm_link_name="rh_palm_r",
    fingertip_link_names=["rh_thdistal_r", "rh_ffdistal_r", "rh_mfdistal_r", "rh_rfdistal_r", "rh_lfdistal_r"],
)
