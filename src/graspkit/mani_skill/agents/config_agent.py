"""ManiSkill agents from a `RobotConfig` for robots without a hand-written agent (floating hands, arm+hand rigs)."""

import numpy as np
import sapien
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import REGISTERED_AGENTS, register_agent
from mani_skill.sensors.camera import CameraConfig

from graspkit.configs.env_config import CameraSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.mani_skill.agents.grasp_agent import GraspAgent
from graspkit.mani_skill.envs.dex_grasp import MsDexGraspEnv
from graspkit.utils.urdf_utils import export_sim_urdf


def register_config_agent(config: RobotConfig, cameras: tuple[CameraSpec, ...] = ()) -> str:
    """Register (once) and return uid `<name>[_floating_base][_cameras]`; a hand-written agent under that uid wins."""
    floating = config.base_type == "floating"
    uid = f"{config.name}_floating_base" if floating else config.name
    if cameras:
        uid += "_cameras"
    if uid in REGISTERED_AGENTS:
        return uid
    urdf, hand_joints = export_sim_urdf(config)
    tips = list(config.fingertip_link_names)

    class ConfigAgent(GraspAgent):
        fix_root_link = True
        load_multiple_collisions = True
        disable_self_collisions = True
        contact_link_names = tips
        palm_link_name = config.palm_link_name

        @property
        def _sensor_configs(self) -> list[CameraConfig]:
            return [
                CameraConfig(
                    camera.name,
                    sapien.Pose(p=camera.pose[:3], q=camera.pose[3:]),
                    width=camera.width,
                    height=camera.height,
                    fov=None if camera.fov is None else float(np.deg2rad(camera.fov)),
                    intrinsic=None if camera.fov is not None else np.asarray(camera.intrinsic),
                    near=0.005,
                    far=5,
                    entity_uid=camera.mount,
                )
                for camera in cameras
            ]

        @property
        def _controller_configs(self):
            names = [j.name for j in self.robot.active_joints]
            hand = PDJointPosControllerConfig(
                [n for n in names if n in hand_joints],
                stiffness=1e3, damping=50, force_limit=50, friction=0.05,
                normalize_action=False, lower=None, upper=None, use_delta=False,
            )  # fmt: skip
            if floating:
                return {"pd_joint_pos": {"floating_base": self.floating_base_controller(), "hand_joints": hand}}
            arm = PDJointPosControllerConfig(
                [n for n in names if n not in hand_joints],
                stiffness=1e6, damping=1e5, force_limit=500,
                normalize_action=False, lower=None, upper=None, use_delta=False,
            )  # fmt: skip
            return {"pd_joint_pos": {"arm_joints": arm, "hand_joints": hand}}

    ConfigAgent.uid = uid
    ConfigAgent.urdf_config = {
        "_materials": {"tip": {"static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.0}},
        "link": {name: {"material": "tip", "patch_radius": 0.05, "min_patch_radius": 0.04} for name in tips},
    }
    ConfigAgent.urdf_path = str(urdf)
    register_agent()(ConfigAgent)
    MsDexGraspEnv.SUPPORTED_ROBOTS.append(uid)
    return uid
