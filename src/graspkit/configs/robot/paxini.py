from robokit.assets.robots.end_effectors import paxini_hand as paxini_assets

from graspkit.configs.robot_config import RobotConfig

# Contacts span the proximal/middle/distal links and palm; the published fingertip-only points made every grasp a pinch.
PAXINI_CONTACTS = {f"right_{f}_link_{i}": 10 for f in ("index", "middle", "ring", "thumb") for i in (1, 2, 3)} | {
    "right_palm_link": 20
}

# Paxini DexH13, four 4-DoF fingers; per URDF FK, index/middle/ring reach +z at rest and flex toward +x (palm side).
paxini_right = RobotConfig(
    name="paxini_right",
    urdf_or_mjcf_path=str(paxini_assets.URDF_MERGED_PATH_RIGHT),
    base_type="floating",
    ee_coord_spec=paxini_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=paxini_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(paxini_assets.COLLISION_SPHERE_PATH_RIGHT),
    contact_candidates_config=PAXINI_CONTACTS,
    self_collision_ignore_path=str(paxini_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    init_qpos=(0.0,) * 16,
    palm_link_name="right_palm_link",
    thumb_link_names=["right_thumb_link_0", "right_thumb_link_1", "right_thumb_link_2", "right_thumb_link_3"],
    middle_link_names=["right_middle_link_0", "right_middle_link_1", "right_middle_link_2", "right_middle_link_3"],
    fingertip_link_names=[
        "right_thumb_link_3",
        "right_index_link_3",
        "right_middle_link_3",
        "right_ring_link_3",
    ],
)
