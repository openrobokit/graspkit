from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class AllegroRightFloatingBase(GraspAgent):
    uid = "allegro_right_floating_base"

    disable_self_collisions: bool = True
    urdf_config = dict(
        _materials=dict(tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)),
        link={
            "link_15.0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_15.0": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_3.0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_3.0": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_7.0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_7.0": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_11.0_tip": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "link_11.0": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "base_link"
        self.contact_link_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import allegro_hand as allegro_hand_assets

        return str(allegro_hand_assets.URDF_GLB_FLOATING_BASE_PATH_RIGHT)

    @property
    def _controller_configs(self):
        joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            stiffness=150,
            damping=10,
            friction=0.05,
            normalize_action=False,
            force_limit=10,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {"pd_joint_pos": joint_pos}
