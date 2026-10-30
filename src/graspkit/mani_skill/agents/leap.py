import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class LeapRightFloatingBase(GraspAgent):
    uid = "leap_right_floating_base"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link=dict(
            thumb_dip=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            thumb_fingertip=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            pip=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            dip=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            fingertip=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            pip_2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            dip_2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            fingertip_2=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            pip_3=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            dip_3=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            fingertip_3=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
        ),
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "palm_lower"
        self.contact_link_names = [
            "thumb_dip",
            "thumb_fingertip",
            "pip",
            "dip",
            "fingertip",
            "pip_2",
            "dip_2",
            "fingertip_2",
            "pip_3",
            "dip_3",
            "fingertip_3",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import leap_hand as leap_hand_assets

        return str(leap_hand_assets.URDF_GLB_FLOATING_BASE_PATH_RIGHT)

    @property
    def _controller_configs(self):
        floating_base = self.floating_base_controller()
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[6:]],
            stiffness=500,
            damping=50,
            normalize_action=False,
            friction=0.05,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {
            "pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos},
        }
