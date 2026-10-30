import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class AbilityRightFloatingBase(GraspAgent):
    uid = "ability_right_floating_base"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link=dict(
            thumb_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            index_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            middle_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            ring_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            pinky_L2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
        ),
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "base"
        self.contact_link_names = ["thumb_L2", "index_L2", "middle_L2", "ring_L2", "pinky_L2"]
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.3))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import ability_hand as ability_hand_assets

        return str(ability_hand_assets.URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_RIGHT)

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
        return {
            "pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos},
        }


@register_agent()
class AbilityLeftFloatingBase(GraspAgent):
    uid = "ability_left_floating_base"

    disable_self_collisions: bool = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "hand_base_link"
        self.contact_link_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import ability_hand as ability_hand_assets

        return str(ability_hand_assets.URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_LEFT)

    @property
    def _controller_configs(self):
        name = [x.name for x in self.robot.active_joints]

        floating_base = self.floating_base_controller()

        joint_pos = PDJointPosControllerConfig(
            name[6:],
            stiffness=150,
            damping=10,
            friction=0.05,
            normalize_action=False,
            force_limit=10,
            lower=None,
            upper=None,
            use_delta=False,
        )

        return {
            "pd_joint_pos": dict(floating_base=floating_base, joint_pos=joint_pos),
        }
