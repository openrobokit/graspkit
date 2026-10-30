from robokit.assets.robots.end_effectors import orcahand as orca_assets

from graspkit.configs.robot_config import RobotConfig

# ORCA v1, 17 DoF; `*_fingertip` links are massless frames, so tip geometry is on `*_ip` (fingers) and `thumb_dp`.
ORCA_TIPS = ["right_thumb_dp", "right_index_ip", "right_middle_ip", "right_ring_ip", "right_pinky_ip"]
# Contacts span the proximal/middle/distal links and palm; the published fingertip-only points made every grasp a pinch.
ORCA_CONTACTS = {f"right_{f}_{s}": 10 for f in ("index", "middle", "ring", "pinky") for s in ("pp", "ip")} | {
    "right_thumb_pp": 10,
    "right_thumb_ip": 10,
    "right_thumb_dp": 16,
    "right_palm": 20,
}

orca_right = RobotConfig(
    name="orca_right",
    urdf_or_mjcf_path=str(orca_assets.URDF_PATH_RIGHT),
    base_type="floating",
    ee_coord_spec=orca_assets.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=orca_assets.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(orca_assets.COLLISION_SPHERE_PATH_RIGHT),
    contact_candidates_config=ORCA_CONTACTS,
    self_collision_ignore_path=str(orca_assets.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT),
    init_qpos=(0.0,) * 17,
    palm_link_name="right_palm",
    thumb_link_names=["right_thumb_mp", "right_thumb_pp", "right_thumb_ip", "right_thumb_dp"],
    middle_link_names=["right_middle_mp", "right_middle_pp", "right_middle_ip"],
    fingertip_link_names=ORCA_TIPS,
)
