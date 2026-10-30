import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent

# Right Sharpa hand contact pads (the links that carry collision spheres / friction material).
_TIP_LINKS = [
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
]

# The 22 right-hand actuated joints (URDF order); everything else is a stiff "body" joint held at its frozen value.
_HAND_JOINTS = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


@register_agent()
class VegaSharpaRight(GraspAgent):
    uid = "vega_sharpa_right"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={tip: dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04) for tip in _TIP_LINKS},
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    hand_joints = _HAND_JOINTS
    palm_link_name = "right_hand_C_MC"
    contact_link_names = _TIP_LINKS

    def __init__(self, *args, **kwargs):
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from graspkit.configs.robot.vega_sharpa import vega_sharpa_right

        return vega_sharpa_right.urdf_or_mjcf_path

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
            stiffness=5e2,
            damping=5e1,
            normalize_action=False,
            force_limit=20,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return dict(pd_joint_pos={"body_joints": body_joint_pos, "hand_joints": hand_joint_pos})


_TIP_LINKS_LEFT = [n.replace("right_", "left_") for n in _TIP_LINKS]


@register_agent()
class VegaSharpaLeft(VegaSharpaRight):
    uid = "vega_sharpa_left"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={tip: dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04) for tip in _TIP_LINKS_LEFT},
    )
    hand_joints = [n.replace("right_", "left_") for n in _HAND_JOINTS]
    palm_link_name = "left_hand_C_MC"
    contact_link_names = _TIP_LINKS_LEFT
