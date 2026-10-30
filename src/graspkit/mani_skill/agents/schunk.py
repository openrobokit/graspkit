import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class SchunkRightFloatingBase(GraspAgent):
    uid = "schunk_right_floating_base"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link=dict(
            right_hand_b=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_a=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_c=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_l=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_p=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_t=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_k=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_o=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_s=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_j=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_n=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_r=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_i=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_m=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
            right_hand_q=dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04),
        ),
    )
    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "right_hand_e1"
        self.contact_link_names = [
            "right_hand_c",
            "right_hand_t",
            "right_hand_s",
            "right_hand_r",
            "right_hand_b",
            "right_hand_p",
            "right_hand_o",
            "right_hand_n",
            "right_hand_m",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0, 0, 0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.end_effectors import schunk_hand as schunk_hand_assets

        return str(schunk_hand_assets.URDF_GLB_FLOATING_BASE_NO_MIMIC_PATH_RIGHT)

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
