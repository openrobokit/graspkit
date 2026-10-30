import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class ManoRightFloatingBase(GraspAgent):
    uid = "mano_right_floating_base"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link=dict(
            link_0=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_1=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_2=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_3=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_4=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_5=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_6=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_7=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_8=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_9=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_10=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_11=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_12=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_13=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_14=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_15=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_16_tip=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_17_tip=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_18_tip=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_19_tip=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            link_20_tip=dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "link_0"
        self.contact_link_names = [
            "link_3",
            "link_6",
            "link_9",
            "link_12",
            "link_15",
            "link_14",
            "link_2",
            "link_5",
            "link_8",
            "link_11",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import mano_hand as mano_hand_assets

        return str(mano_hand_assets.URDF_FLOATING_BASE_PATH_RIGHT)

    @property
    def _controller_configs(self):
        floating_base = self.floating_base_controller()
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[6:]],
            stiffness=2e4,
            damping=100,
            normalize_action=False,
            friction=0.5,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {
            "pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos},
        }
