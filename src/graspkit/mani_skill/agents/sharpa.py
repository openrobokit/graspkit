import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent

_TIP_LINKS = [
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
]


@register_agent()
class SharpaRightFloatingBase(GraspAgent):
    uid = "sharpa_right_floating_base"
    palm_link_name = "right_hand_C_MC"
    contact_link_names = _TIP_LINKS

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={tip: dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04) for tip in _TIP_LINKS},
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.3))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import sharpa_hand as sharpa_hand_assets

        return str(sharpa_hand_assets.URDF_FLOATING_BASE_PATH_RIGHT)

    @property
    def _controller_configs(self):
        floating_base = self.floating_base_controller()
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[6:]],
            stiffness=500,
            damping=50,
            normalize_action=False,
            force_limit=20,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {"pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos}}


@register_agent()
class SharpaLeftFloatingBase(SharpaRightFloatingBase):
    uid = "sharpa_left_floating_base"
    palm_link_name = "left_hand_C_MC"
    contact_link_names = [name.replace("right_", "left_") for name in _TIP_LINKS]

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={
            name.replace("right_", "left_"): dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04)
            for name in _TIP_LINKS
        },
    )

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import sharpa_hand as sharpa_hand_assets

        return str(sharpa_hand_assets.URDF_FLOATING_BASE_PATH_LEFT)
