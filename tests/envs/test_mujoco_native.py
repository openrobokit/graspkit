"""Tests for `graspkit.envs.mujoco_native.MjNativeEnv`."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from graspkit.configs.env_config import MjNativeEnvConfig
from graspkit.configs.experiment_config import EnvConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.envs import build_env
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription

LINK = (
    '<link name="{}"><inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>\
</inertial><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>'
)
JOINT = '<joint name="{}" type="prismatic"><parent link="{}"/><child link="{}"/><axis xyz="0 1 0"/>\
<limit lower="-0.2" upper="0.2" effort="1" velocity="1"/>{}</joint>'


class TestMjNativeEnv:
    def test_hold_teleport_and_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A slider and its mimic hold mirrored targets, a cube stays put; a teleported cube is lifted but not grasped."""
        links = "".join(LINK.format(name) for name in ("base", "left", "right"))
        joints = JOINT.format("left_joint", "base", "left", "") + JOINT.format(
            "right_joint", "base", "right", '<mimic joint="left_joint" multiplier="-1" offset="0"/>'
        )
        (tmp_path / "robot.urdf").write_text(f'<robot name="test">{links}{joints}</robot>')
        rig = SimpleNamespace(
            num_actuated_joints=1,
            actuated_joint_names=["left_joint"],
            init_qpos=torch.tensor([0.05]),
            root_pose_7d=torch.tensor([0.0, 0.0, 0.3, 1, 0, 0, 0]),
        )
        full = SimpleNamespace(
            actuated_joint_names=["left_joint", "right_joint"], init_qpos=torch.tensor([0.05, -0.05])
        )
        monkeypatch.setattr(
            "graspkit.isaac.job.GraspRobot.from_config", lambda *args, prune=True, **kw: rig if prune else full
        )
        monkeypatch.setattr("graspkit.utils.urdf_utils.SIM_URDF_ROOT", tmp_path / "sim_urdf")
        trimesh.creation.box(extents=(0.04, 0.04, 0.04)).export(tmp_path / "cube.obj")
        scene = SceneDescription(
            scene_id="cube",
            target_actor_id="object",
            meshes={"object": MeshDescription("object", str(tmp_path / "cube.obj"), (1, 1, 1), position=(0, 0, 0.02))},
            boxes={"table": BoxDescription("table", (0.5, 0.5, 0.1), position=(0, 0, -0.05))},
        )
        robot = RobotConfig(
            name="test", urdf_or_mjcf_path=str(tmp_path / "robot.urdf"), hand_base_link_name="base",
            fingertip_link_names=["left", "right"], base_type="fixed",
        )  # fmt: skip
        env = build_env(EnvConfig(mj_native_env_config=MjNativeEnvConfig(num_threads=2)), robot, "cpu")
        env.reset([scene, scene])
        for _ in range(25):
            env.step_qpos(torch.tensor([[0.05], [0.08]]))
        np.testing.assert_allclose(env.get_qpos().numpy(), [[0.05], [0.08]], atol=1e-3)
        np.testing.assert_allclose(env.datas[1].qpos[env.joints], [0.08, -0.08], atol=1e-3)
        pose = env.get_object_pose()
        np.testing.assert_allclose(pose[:, :3].numpy(), [[0, 0, 0.02]] * 2, atol=1e-3)
        env.set_object_pose(pose + torch.tensor([0, 0, 0.3, 0, 0, 0, 0]))
        info = env.get_info()
        assert info["is_lifted"].all() and not info["is_strict_grasped"].any() and not info["success"].any()
        np.testing.assert_allclose(info["lift_dz"].numpy(), 0.3, atol=2e-3)
        env.close()
