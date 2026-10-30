"""Tests for `graspkit.datasets.graspkit_data`."""

from dataclasses import fields
from pathlib import Path

import h5py
import pytest
import torch

from graspkit.datasets.graspkit_data import GraspKitData


class TestGraspKitData:
    def test_runtime_fields_survive_layout_changes(self):
        qpos = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        traj = torch.arange(48, dtype=torch.float32).reshape(2, 2, 4, 3)
        penetration = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        data = GraspKitData(
            scene_ids=["a", "b"],
            pre_grasp_qpos=qpos,
            grasp_qpos=qpos + 1,
            post_grasp_qpos=qpos + 2,
            full_traj_qpos=traj,
            pre_traj_penetration=penetration,
        )

        packed = data.to_packed()
        assert torch.equal(packed.full_traj_qpos, traj.reshape(4, 4, 3))
        assert torch.equal(packed.pre_traj_penetration, penetration.flatten())

        filtered = packed.filter(torch.tensor([True, False, True, False])).to_padded()
        assert torch.equal(filtered.full_traj_qpos[:, 0], traj[:, 0])
        assert torch.equal(filtered.pre_traj_penetration[:, 0], penetration[:, 0])

    def test_h5_persists_only_live_disk_fields(self, tmp_path: Path):
        qpos = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
        data = GraspKitData(
            scene_ids=["scene"],
            pre_grasp_qpos=qpos,
            grasp_qpos=qpos + 1,
            post_grasp_qpos=qpos + 2,
            full_traj_qpos=torch.zeros(1, 2, 4, 6),
            pre_traj_penetration=torch.zeros(1, 2),
            pointcloud=torch.zeros(1, 1, 2, 3),
            target_actor_segmentation_id=torch.tensor([7]),
        )
        path = tmp_path / "data"
        data.save(str(path))

        extra = GraspKitData(
            scene_ids=["scene"],
            pre_grasp_qpos=qpos[:, :1] + 10,
            grasp_qpos=qpos[:, :1] + 11,
            post_grasp_qpos=qpos[:, :1] + 12,
            full_traj_qpos=torch.ones(1, 1, 8, 6),
            pre_traj_penetration=torch.ones(1, 1),
        )
        extra.save(str(path), append=True)

        h5_path = path / "scene.h5"
        with h5py.File(h5_path, "a") as f:
            assert set(f) == {
                "grasp_qpos",
                "pointcloud",
                "post_grasp_qpos",
                "pre_grasp_qpos",
                "target_actor_segmentation_id",
            }
            f.create_dataset("full_traj_qpos", data=torch.zeros(3, 4, 6).numpy())
            f.create_dataset("pre_traj_penetration", data=torch.zeros(3).numpy())
            f.create_dataset("obs_agent_qpos", data=torch.zeros(1, 4, 6).numpy())
            f.create_dataset("object_poses", data=torch.zeros(1, 4, 7).numpy())
            f.create_dataset("actions", data=torch.zeros(3, 3, 6).numpy())
            f.create_dataset("action_indices", data=torch.zeros(3, dtype=torch.int64).numpy())
            f.create_dataset("best_point_indices", data=torch.zeros(3, dtype=torch.int64).numpy())
            f.create_dataset("grasp_reason_mask", data=torch.zeros(3, dtype=torch.uint8).numpy())
            f.create_dataset("graspness_weights", data=torch.zeros(2).numpy())

        loaded = GraspKitData.load(str(path), packed=True)
        assert torch.equal(loaded.grasp_qpos, torch.cat([qpos[0] + 1, qpos[0, :1] + 11]))
        assert loaded.full_traj_qpos is None
        assert loaded.pre_traj_penetration is None
        removed = {
            "actions",
            "action_indices",
            "best_point_indices",
            "grasp_reason_mask",
            "graspness_weights",
            "object_poses",
            "obs_agent_qpos",
        }
        assert removed.isdisjoint(field.name for field in fields(GraspKitData))

    def test_packed_padded_roundtrip_and_counts(self):
        qpos = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        data = GraspKitData(scene_ids=["a", "b"], grasp_qpos=qpos, pointcloud=torch.zeros(2, 1, 4, 3))
        assert data.is_padded and data.num_grasps_per_scene.tolist() == [3, 3]
        assert GraspKitData(scene_ids=["a", "b"]).num_grasps_per_scene.tolist() == [0, 0]

        packed = data.to_packed()
        assert packed.is_packed and packed.first_indices.tolist() == [0, 3, 6]
        assert packed.grasp_qpos.shape == (6, 3)
        assert packed.to_packed() is packed

        padded = packed.to_padded()
        assert torch.equal(padded.grasp_qpos, qpos)
        assert torch.equal(padded.pointcloud, data.pointcloud)
        assert padded.to_padded() is padded

        ragged = GraspKitData(scene_ids=["a", "b"], first_indices=torch.tensor([0, 1, 3]), grasp_qpos=torch.zeros(3, 3))
        with pytest.raises(ValueError, match="uniform K"):
            ragged.to_padded()

    def test_filter_scene_and_grasp_masks(self):
        qpos = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        data = GraspKitData(
            scene_ids=["a", "b", "c"],
            first_indices=torch.tensor([0, 2, 3, 6]),
            grasp_qpos=qpos,
            pointcloud=torch.arange(9, dtype=torch.float32).reshape(3, 1, 1, 3),
        )

        by_scene = data.filter(torch.tensor([True, False, True]))
        assert by_scene.scene_ids == ["a", "c"]
        assert by_scene.first_indices.tolist() == [0, 2, 5]
        assert torch.equal(by_scene.grasp_qpos, qpos[[0, 1, 3, 4, 5]])
        assert torch.equal(by_scene.pointcloud, data.pointcloud[[0, 2]])

        by_grasp = data.filter(torch.tensor([True, False, False, False, True, True]))
        assert by_grasp.scene_ids == ["a", "c"]  # scene b lost its only grasp and is dropped
        assert by_grasp.first_indices.tolist() == [0, 1, 3]
        assert torch.equal(by_grasp.grasp_qpos, qpos[[0, 4, 5]])

        with pytest.raises(AssertionError, match="expected mask shape"):
            data.filter(torch.tensor([True, False]))
        with pytest.raises(AssertionError, match="packed layout"):
            GraspKitData(scene_ids=["a"], grasp_qpos=torch.zeros(1, 2, 3)).filter(torch.tensor([True]))

    def test_merged_h5_save_load(self, tmp_path: Path):
        qpos = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        data = GraspKitData(scene_ids=["a", "b"], grasp_qpos=qpos, pointcloud=torch.zeros(2, 1, 4, 3))
        path = str(tmp_path / "merged.h5")
        data.save(path)
        data.save(path, append=True)  # grasp rows append, scene fields replace

        loaded = GraspKitData.load(path, packed=True)
        assert loaded.first_indices.tolist() == [0, 4, 8]
        assert torch.equal(loaded.grasp_qpos, qpos.repeat(1, 2, 1).reshape(8, 3))
        assert loaded.pointcloud.shape == (2, 1, 4, 3)

        subset = GraspKitData.load(path, scene_ids=["b"], packed=False)
        assert subset.scene_ids == ["b"] and subset.grasp_qpos.shape == (1, 4, 3)

        assert GraspKitData.load(str(tmp_path / "missing"), packed=True).num_scenes == 0
