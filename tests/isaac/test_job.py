"""Tests for `graspkit.isaac.job`."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import trimesh

from graspkit.configs.env_config import CameraSpec
from graspkit.configs.robot_config import RobotConfig
from graspkit.isaac.job import prepare_robot, write_job
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription


class TestJob:
    def test_scene_job_export(self, tmp_path: Path):
        mesh_path = tmp_path / "cube.obj"
        trimesh.creation.box(extents=(0.04, 0.04, 0.04)).export(mesh_path)
        scene = SceneDescription(
            scene_id="cube",
            target_actor_id="object",
            meshes={"object": MeshDescription("object", str(mesh_path), (2, 3, 4), position=(0, 0, 0.05))},
            boxes={"table": BoxDescription("table", (1, 1, 0.1), position=(0, 0, -0.05))},
        )
        qpos = np.asarray([[[0.1, 0.2, -0.15], [0.3, 0.4, -0.55]]], np.float32)
        job_path = tmp_path / "job.h5"
        write_job(job_path, [scene], qpos, np.asarray([0, 0, 0, 1, 0, 0, 0]), {"control_freq": 50})
        with h5py.File(job_path) as job:
            np.testing.assert_allclose(job["qpos"][...], qpos)
            assert job["scene_id"][0].decode() == "cube" and job.attrs["control_freq"] == 50
            np.testing.assert_allclose(np.ptp(job["meshes/0/0/vertices"][...], axis=0), [0.08, 0.12, 0.16])
            np.testing.assert_allclose(np.ptp(job["visuals/0/vertices"][...], axis=0), [0.08, 0.12, 0.16])
            assert job["visuals/0/colors"].shape == (len(job["visuals/0/vertices"]), 3)
            np.testing.assert_array_equal(job["root_pose"][0], [0, 0, 0, 1, 0, 0, 0])

    def test_robot_mimics_cameras_and_native_appearance(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        urdf = """<robot name="test">
      <link name="base"/><link name="arm"/><link name="finger"/><link name="mimic"/><link name="other"/>
      <joint name="arm_joint" type="revolute"><parent link="base"/><child link="arm"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="finger_joint" type="revolute"><parent link="arm"/><child link="finger"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="mimic_joint" type="revolute"><parent link="finger"/><child link="mimic"/><limit lower="-1" upper="1" effort="1" velocity="1"/><mimic joint="finger_joint" multiplier="-2" offset="0.25"/></joint>
      <joint name="other_joint" type="revolute"><parent link="base"/><child link="other"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
    </robot>"""
        (tmp_path / "source.urdf").write_text(urdf)
        rig = SimpleNamespace(
            num_actuated_joints=2,
            actuated_joint_names=["arm_joint", "finger_joint"],
            init_qpos=torch.tensor([0.1, 0.2]),
            root_pose_7d=torch.tensor([0, 0, 0, 1, 0, 0, 0]),
        )
        full = SimpleNamespace(
            actuated_joint_names=["arm_joint", "finger_joint", "other_joint"], init_qpos=torch.tensor([0.1, 0.2, 0.7])
        )
        monkeypatch.setattr(
            "graspkit.isaac.job.GraspRobot.from_config", lambda *args, prune=True, **kwargs: rig if prune else full
        )
        monkeypatch.setattr("graspkit.utils.urdf_utils.SIM_URDF_ROOT", tmp_path / "sim_urdf")
        appearance = tmp_path / "appearance"
        (appearance / "native_robot_materials").mkdir(parents=True)
        (appearance / "native_appearance").mkdir()
        contract = {
            "robots": {"fixture": {"robot_config_name": "test", "effective_native_material_getters": [{"x": 1}]}}
        }
        (appearance / "native_robot_materials/contract.json").write_text(json.dumps(contract))
        with h5py.File(appearance / "native_appearance/table_visual.h5", "w") as table:
            table["textures/0/png"] = np.frombuffer(b"png", np.uint8)
        config = RobotConfig(
            name="test",
            urdf_or_mjcf_path=str(tmp_path / "source.urdf"),
            hand_base_link_name="finger",
            palm_link_name="arm",
            fingertip_link_names=["finger"],
        )
        robot = prepare_robot(config, (CameraSpec("cam", "arm", (0, 0, 1, 1, 0, 0, 0), fov=90),), tmp_path, appearance)
        np.testing.assert_allclose(
            np.asarray([[0.3, 0.4]]) @ robot["coefficients"].T + robot["bias"], [[0.3, 0.4, -0.55, 0.7]]
        )
        attrs = robot["attrs"]
        assert attrs["joint_names"] == ["arm_joint", "finger_joint", "mimic_joint", "other_joint"]
        assert attrs["policy_joint_names"] == ["arm_joint", "finger_joint"]
        assert attrs["hand_joints"] == ["finger_joint", "mimic_joint"]
        assert attrs["contact_links"] == ["finger", "arm"] and attrs["tip_links"] == ["finger"]
        camera = json.loads(attrs["cameras"])[0]
        assert camera["mount"] == "arm" and camera["pose"] == [0, 0, 1, 1, 0, 0, 0]
        np.testing.assert_allclose(camera["intrinsic"], [[120, 0, 160], [0, 120, 120], [0, 0, 1]])
        assert json.loads(attrs["materials"]) == [{"x": 1}] and json.loads(attrs["camera_materials"]) == {}
        assert Path(json.loads(attrs["table_textures"])["0"]).read_bytes() == b"png"
        assert all(link.find("inertial") is not None for link in ET.parse(attrs["urdf"]).getroot().iter("link"))
