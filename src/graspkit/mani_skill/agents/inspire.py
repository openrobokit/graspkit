import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class InspireRightFloatingBase(GraspAgent):
    uid = "inspire_right_floating_base"

    urdf_config = dict(
        _materials=dict(tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link={
            "index_intermediate": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "middle_intermediate": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "ring_intermediate": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "pinky_intermediate": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "thumb_distal": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "thumb_intermediate": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
        },
    )
    disable_self_collisions: bool = True
    fix_root_link: bool = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "hand_base_link"
        self.contact_link_names = [
            "thumb_distal",
            "index_intermediate",
            "middle_intermediate",
            "ring_intermediate",
            "pinky_intermediate",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import inspire_hand as inspire_hand_assets

        return str(inspire_hand_assets.URDF_FLOATING_BASE_NO_MIMIC_PATH_RIGHT)

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
