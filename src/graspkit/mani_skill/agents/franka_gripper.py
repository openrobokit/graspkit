import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

from graspkit.mani_skill.agents.grasp_agent import GraspAgent


@register_agent()
class FrankaGripper(GraspAgent):
    uid = "franka_gripper"
    contact_link_names = ["panda_leftfinger", "panda_rightfinger"]
    palm_link_name = "panda_hand"

    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(
            panda_leftfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            panda_rightfinger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.arms import franka_panda as franka_panda_assets

        return str(franka_panda_assets.URDF_NO_MIMIC_PATH)

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = dict(pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos))

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_leftfinger")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_rightfinger")
        self.finger1pad_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_leftfinger_pad")
        self.finger2pad_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "panda_rightfinger_pad")
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)


@register_agent()
class FrankaGripperFloatingBase(GraspAgent):
    uid = "franka_gripper_floating_base"

    urdf_config = dict(
        _materials=dict(tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link={
            "panda_leftfinger": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
            "panda_rightfinger": dict(material="tip", patch_radius=0.05, min_patch_radius=0.04),
        },
    )
    disable_self_collisions: bool = True

    def __init__(self, *args, **kwargs):
        self.palm_link_name = "panda_hand"
        self.contact_link_names = [
            "panda_leftfinger",
            "panda_rightfinger",
        ]
        kwargs["initial_pose"] = sapien.Pose(p=(0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    @property
    def urdf_path(self) -> str:
        from robokit.assets.robots.arms import franka_panda as franka_panda_assets

        return str(franka_panda_assets.GRIPPER_NO_MIMIC_URDF_PATH)

    @property
    def _controller_configs(self):
        floating_base = self.floating_base_controller()
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[6:]],
            stiffness=1000,
            damping=50,
            normalize_action=False,
            friction=0.05,
            lower=None,
            upper=None,
            use_delta=False,
        )
        joint_pos_for_planning = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            stiffness=1e3,
            damping=1e2,
            normalize_action=False,
            friction=0.05,
            lower=None,
            upper=None,
            use_delta=False,
        )
        joint_pos_delta = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints],
            lower=-0.1,
            upper=0.1,
            stiffness=1e3,
            damping=0.0,
            friction=0.05,
            normalize_action=True,
            use_delta=True,
        )
        return {
            "pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos},
            "pd_joint_pos_for_planning": joint_pos_for_planning,
            "pd_joint_delta_pos": joint_pos_delta,
        }
