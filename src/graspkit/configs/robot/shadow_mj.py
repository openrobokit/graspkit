from robokit.assets.robots.end_effectors import shadow_hand as shadow_hand_assets

from graspkit.configs.robot_config import RobotConfig

# Contact candidates sampled on the middle/distal finger links + thumb (thumb weighted higher).
SHADOW_MJ_CONTACT_CANDIDATES = {
    "rh_thproximal": 5,
    "rh_thmiddle": 10,
    "rh_thdistal": 20,
    "rh_ffmiddle": 5,
    "rh_ffdistal": 10,
    "rh_mfmiddle": 5,
    "rh_mfdistal": 10,
    "rh_rfmiddle": 5,
    "rh_rfdistal": 10,
    "rh_lfmiddle": 5,
    "rh_lfdistal": 10,
}

SHADOW_MJ_XML = str(shadow_hand_assets.MJ_DESCRIPTION_DIR_RIGHT / "right_hand.xml")

# DexGraspBench menagerie Shadow hand -- same model MjDexGraspEnv evaluates (no remap); FK-verified palm normal is -y.
shadow_right_mj = RobotConfig(
    name="shadow_right_mj",
    urdf_or_mjcf_path=SHADOW_MJ_XML,
    base_type="floating",
    keep_world_link=True,  # root the floating base at the mjcf worldbody = MjDexGraspEnv's freejoint frame
    init_qpos=(0.0,) * 22,
    ee_coord_spec=shadow_hand_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name="rh_palm",  # the MJCF has no forearm; its palm is oriented like the URDF forearm,
    collision_spheres_path=str(shadow_hand_assets.MJ_COLLISION_SPHERE_PATH_RIGHT),
    contact_candidates_config=SHADOW_MJ_CONTACT_CANDIDATES,
    thumb_link_names=["rh_thproximal", "rh_thhub", "rh_thmiddle", "rh_thdistal"],
    middle_link_names=["rh_mfknuckle", "rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"],
    palm_link_name="rh_palm",
    fingertip_link_names=["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"],
)
