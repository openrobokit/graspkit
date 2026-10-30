import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class Xarm7AbilityRight(GraspAgent):
    uid = "xarm7_ability_right"

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

    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    hand_joints = [
        "thumb_q1",
        "index_q1",
        "middle_q1",
        "ring_q1",
        "pinky_q1",
        "thumb_q2",
        "index_q2",
        "middle_q2",
        "ring_q2",
        "pinky_q2",
    ]
    palm_link_name = "base"
    contact_link_names = ["thumb_L2", "index_L2", "middle_L2", "ring_L2", "pinky_L2"]

    def __init__(self, *args, **kwargs):
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.assembly import xarm7_ability as xarm7_ability_assets

        return str(xarm7_ability_assets.URDF_NO_MIMIC_PATH)

    def _after_init(self):
        pass

    @property
    def _controller_configs(self):
        arm_joint_pos = PDJointPosControllerConfig(
            self.arm_joints,
            stiffness=1e6,
            damping=1e5,
            normalize_action=False,
            force_limit=500,
            lower=None,
            upper=None,
            use_delta=False,
        )
        hand_joint_pos = PDJointPosControllerConfig(
            self.hand_joints,
            stiffness=1e1,
            damping=1e0,
            normalize_action=False,
            force_limit=50,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {
            "pd_joint_pos": {
                "arm_joints": arm_joint_pos,
                "hand_joints": hand_joint_pos,
            },
        }
