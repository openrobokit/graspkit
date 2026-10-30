import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class Xarm7XHandRight(GraspAgent):
    uid = "xarm7_xhand_right"

    urdf_config = dict(
        _materials=dict(finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link={
            "right_hand_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_back_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_bend_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_rota_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_rotaback_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_rota_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_rotaback_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_thumb_rota_tip": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_bend_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_rota_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_rotaback_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_rota_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_rotaback_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_index_rota_tip": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_mid_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_midback_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_mid_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_midback_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_mid_tip": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_ring_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_ringback_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_ring_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_ringback_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_ring_tip": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_pinky_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_pinkyback_link1": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_pinky_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_pinkyback_link2": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "right_hand_pinky_tip": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    hand_joints = [
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
    ]

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "right_hand_link"
        self.contact_link_names = [
            "right_hand_thumb_rota_link2",
            "right_hand_thumb_rota_tip",
            "right_hand_index_rota_link2",
            "right_hand_index_rota_tip",
            "right_hand_mid_link2",
            "right_hand_mid_tip",
            "right_hand_ring_link2",
            "right_hand_ring_tip",
            "right_hand_pinky_link2",
            "right_hand_pinky_tip",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.assembly import xarm7_xhand as xarm7_xhand_assets

        return str(xarm7_xhand_assets.URDF_PATH)

    @property
    def _controller_configs(self):
        arm_joint_pos = PDJointPosControllerConfig(
            self.arm_joints,
            stiffness=1e6,
            damping=1e5,
            normalize_action=False,
            force_limit=500,
            lower=None,
            upper=None,
            use_delta=False,
        )
        hand_joint_pos = PDJointPosControllerConfig(
            [j.name for j in self.robot.active_joints if j.name in self.hand_joints],
            stiffness=1e3,
            damping=50,
            normalize_action=False,
            force_limit=50,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {
            "pd_joint_pos": {
                "arm_joints": arm_joint_pos,
                "hand_joints": hand_joint_pos,
            },
        }
