import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class Xarm7ManoRight(GraspAgent):
    uid = "xarm7_mano_right"

    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={
            "mano_link_0": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_1": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_2": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_3": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_4": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_5": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_6": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_7": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_8": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_9": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_10": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_11": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_12": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_13": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_14": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_15": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_16_tip": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_17_tip": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_18_tip": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_19_tip": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
            "mano_link_20_tip": dict(material="front_finger", patch_radius=0.1, min_patch_radius=0.1),
        },
    )

    fix_root_link = True
    load_multiple_collisions = True
    disable_self_collisions = True

    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    hand_joints = [
        "mano_joint_1",
        "mano_joint_2",
        "mano_joint_3",
        "mano_joint_4",
        "mano_joint_5",
        "mano_joint_6",
        "mano_joint_7",
        "mano_joint_8",
        "mano_joint_9",
        "mano_joint_10",
        "mano_joint_11",
        "mano_joint_12",
        "mano_joint_empty",
        "mano_joint_13",
        "mano_joint_14",
        "mano_joint_15",
    ]
    palm_link_name = "mano_link_0"
    contact_link_names = [
        "mano_link_3",
        "mano_link_6",
        "mano_link_9",
        "mano_link_12",
        "mano_link_15",
        "mano_link_14",
        "mano_link_2",
        "mano_link_5",
        "mano_link_8",
        "mano_link_11",
    ]

    def __init__(self, *args, **kwargs):
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.assembly import xarm7_mano as xarm7_mano_assets

        return str(xarm7_mano_assets.URDF_PATH)

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
            stiffness=500,
            damping=50,
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
