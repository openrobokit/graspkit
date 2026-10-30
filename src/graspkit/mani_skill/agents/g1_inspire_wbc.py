import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class G1InspireRightWBC(GraspAgent):
    uid = "g1_inspire_right_wbc"

    urdf_config = dict(
        _materials=dict(finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link={
            "L_hand_base_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_thumb_proximal_base": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_thumb_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_thumb_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_thumb_distal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_index_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_index_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_middle_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_middle_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_ring_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_ring_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_pinky_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "L_pinky_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_hand_base_link": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_thumb_proximal_base": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_thumb_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_thumb_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_thumb_distal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_index_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_index_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_middle_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_middle_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_ring_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_ring_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_pinky_proximal": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
            "R_pinky_intermediate": dict(material="finger", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    hand_joints = [
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "L_thumb_distal_joint",
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint",
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
    ]

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "R_hand_base_link"
        self.contact_link_names = [
            "R_thumb_distal",
            "R_thumb_proximal",
            "R_thumb_intermediate",
            "R_index_intermediate",
            "R_index_proximal",
            "R_middle_intermediate",
            "R_middle_proximal",
            "R_ring_intermediate",
            "R_ring_proximal",
            "R_pinky_proximal",
            "R_pinky_intermediate",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.assembly import g1_inspire as g1_inspire_assets

        return str(g1_inspire_assets.URDF_UPPER_HEAD_NO_MIMIC_PATH)

    @property
    def _controller_configs(self):
        body_joint_pos = PDJointPosControllerConfig(
            [j.name for j in self.robot.active_joints if j.name not in self.hand_joints],
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
            stiffness=2e2,
            damping=2e1,
            normalize_action=False,
            force_limit=100,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return dict(pd_joint_pos={"body_joints": body_joint_pos, "hand_joints": hand_joint_pos})
