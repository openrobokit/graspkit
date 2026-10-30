from robokit.assets.robots.assembly import xarm6_sharpa as xarm6_sharpa_assets
from robokit.assets.robots.end_effectors import sharpa_hand

from graspkit.configs.robot.sharpa import SHARPA_CONTACT_CANDIDATES
from graspkit.configs.robot_config import RobotConfig

# xArm6 (6 dof) + Sharpa (22 dof) = 28 actuated joints, URDF order joint1..joint6 then Sharpa fingers.
xarm6_sharpa_right = RobotConfig(
    name="xarm6_sharpa_right",
    urdf_or_mjcf_path=str(xarm6_sharpa_assets.URDF_PATH),
    self_collision_ignore_path=str(xarm6_sharpa_assets.SPHERE_SELF_COLLISION_IGNORE_PATH),
    base_type="fixed",
    workspace_origin=(-0.41, -0.37, 0.01),
    init_position=(-0.41, -0.37, 0.01),
    init_quaternion=(1.0, 0.0, 0.0, 0.0),
    hand_base_link_name="right_hand_C_MC",
    ee_coord_spec=sharpa_hand.EE_COORD_SPEC_RIGHT,
    ee_coord_link_name=sharpa_hand.MOUNT_LINK_NAME_RIGHT,
    collision_spheres_path=str(xarm6_sharpa_assets.COLLISION_SPHERE_PATH),
    contact_candidates_config=SHARPA_CONTACT_CANDIDATES,
    thumb_link_names=["right_thumb_PP", "right_thumb_DP", "right_thumb_fingertip"],
    middle_link_names=[
        "right_middle_PP",
        "right_middle_MP",
        "right_middle_DP",
        "right_middle_elastomer",
        "right_middle_fingertip",
    ],
    palm_link_name="right_hand_C_MC",
    fingertip_link_names=[  # the *_fingertip links are empty frames; the elastomer pads carry the spheres
        "right_thumb_elastomer",
        "right_index_elastomer",
        "right_middle_elastomer",
        "right_ring_elastomer",
        "right_pinky_elastomer",
    ],
    # arm home pose is a placeholder — tune for xArm6 reachability/clearance before real deploy.
    init_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) + (0.1,) * 22,
)
