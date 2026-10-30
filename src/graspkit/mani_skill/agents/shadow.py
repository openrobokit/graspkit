from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class ShadowRightFloatingBase(GraspAgent):
    uid = "shadow_right_floating_base"

    disable_self_collisions: bool = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "palm"
        self.contact_link_names = ["fftip", "mftip", "rftip", "lftip", "thtip"]

        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import shadow_hand as shadow_hand_assets

        return str(shadow_hand_assets.URDF_FLOATING_BASE_PATH_RIGHT)

    @property
    def _controller_configs(self):
        joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            stiffness=1e3,
            damping=0.0,
            friction=0.05,
            normalize_action=False,
            force_limit=0.5,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {"pd_joint_pos": joint_pos}
