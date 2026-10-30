"""Tests for `graspkit.datasets.grasp_dataset`."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from graspkit.configs.experiment_config import DatasetConfig
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.scene.scene_description import SceneDescription


class TestGraspDataset:
    @pytest.mark.parametrize("aug_prob", [0.0, 1.0])
    def test_cached_fused_views_preserve_augmentation_and_head_rgb(self, tmp_path: Path, aug_prob: float):
        h5, scenes = tmp_path / "views.h5", tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack({"scene": SceneDescription(scene_id="scene")}, str(scenes))
        rng = np.random.default_rng(0)
        with h5py.File(h5, "w") as output:
            group = output.create_group("data/scene")
            obs = group.create_group("obs")
            points = rng.uniform((-0.03, -0.03, 0.03), (0.03, 0.03, 0.08), (3, 120, 3)).astype(np.float32)
            obs["pointcloud"] = points
            obs["pointcloud_segmentation"] = np.broadcast_to((np.arange(120) % 2)[None, :, None], (3, 120, 1))
            obs["target_actor_segmentation_id"] = 1
            obs["camera_poses"] = np.tile([-0.5, 0, 0.1, 1, 0, 0, 0], (3, 1)).astype(np.float32)
            obs["rgb"] = np.stack([np.full((4, 4, 3), value, np.uint8) for value in (11, 22, 33)])
            obs["qpos"] = np.zeros(8, np.float32)
            group["actions"] = rng.normal(size=(4, 3, 8)).astype(np.float32)
        cfg = DatasetConfig(
            num_points=16,
            num_actions=2,
            return_object_pointcloud_only=True,
            fuse_views=2,
            aug_prob=aug_prob,
            view_noise=0.003,
            point_noise=0.0005,
        )
        original = GraspDataset(cfg, str(h5), str(scenes))
        cached = GraspDataset(replace(cfg, cache_in_memory=True), str(h5), str(scenes))
        cached.h5_file.close()
        for seed in range(4):
            np.random.seed(seed)
            expected = original[0]
            expected_rng = np.random.random(8)
            np.random.seed(seed)
            actual = cached[0]
            np.testing.assert_array_equal(np.random.random(8), expected_rng)
            torch.testing.assert_close(actual["actions"], expected["actions"], atol=0, rtol=0)
            for key in expected["observations"]:
                torch.testing.assert_close(actual["observations"][key], expected["observations"][key], atol=0, rtol=0)
            assert (actual["observations"]["rgb"] == 11).all()
        original.h5_file.close()

    @pytest.mark.parametrize("num_points", [8, 10, 16])
    @pytest.mark.parametrize("num_actions", [2, 8])
    @pytest.mark.parametrize("views", [1, 3])
    @pytest.mark.parametrize("object_only", [False, True])
    def test_cached_sampling_is_identical(
        self, tmp_path: Path, num_points: int, num_actions: int, views: int, object_only: bool
    ):
        h5 = tmp_path / "data.h5"
        scenes = tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack({"scene": SceneDescription(scene_id="scene")}, str(scenes))
        with h5py.File(h5, "w") as output:
            group = output.create_group("data/scene")
            obs = group.create_group("obs")
            obs["pointcloud"] = np.arange(90 * views, dtype=np.float32).reshape(views, 30, 3)
            obs["pointcloud_segmentation"] = np.broadcast_to((np.arange(30) % 3).reshape(1, 30, 1), (views, 30, 1))
            obs["target_actor_segmentation_id"] = 2
            group["actions"] = np.arange(4 * 3 * 28, dtype=np.float32).reshape(4, 3, 28)
        cfg = DatasetConfig(num_points=num_points, num_actions=num_actions, return_object_pointcloud_only=object_only)
        original = GraspDataset(cfg, str(h5), str(scenes))
        cached = GraspDataset(replace(cfg, cache_in_memory=True), str(h5), str(scenes))
        cached.h5_file.close()
        for seed in range(4):
            np.random.seed(seed)
            expected = original[0]
            expected_rng = np.random.random(8)
            np.random.seed(seed)
            actual = cached[0]
            np.testing.assert_array_equal(np.random.random(8), expected_rng)
            assert actual["metas"] == expected["metas"]
            torch.testing.assert_close(actual["actions"], expected["actions"], atol=0, rtol=0)
            for key in expected["observations"]:
                torch.testing.assert_close(actual["observations"][key], expected["observations"][key], atol=0, rtol=0)
        original.h5_file.close()

    @pytest.mark.parametrize("cache", [False, True])
    def test_prepared_views_match_device_batches(self, tmp_path: Path, cache: bool):
        h5, scenes = tmp_path / "prepared.h5", tmp_path / "scenes.msgpack"
        ids = ["a", "a__pre", "b"]
        SceneDescription.save_scene_descriptions_msgpack(
            {sid: SceneDescription(scene_id=sid) for sid in ids}, str(scenes)
        )
        with h5py.File(h5, "w") as output:
            for index, sid in enumerate(ids):
                group = output.create_group(f"data/{sid}")
                group["actions"] = np.arange((index + 2) * 3 * 8, dtype=np.float32).reshape(index + 2, 3, 8)
                if sid == "a__pre":
                    group["obs"] = output["data/a/obs"]
                    continue
                obs = group.create_group("obs")
                obs["pointcloud"] = (np.arange(3 * 4 * 3).reshape(3, 4, 3) / 1000).astype(np.float16)
                obs["pointcloud_center"] = np.full((3, 3), index / 10, dtype=np.float32)
                obs.create_dataset("pointcloud_segmentation", shape=(3, 4, 1), dtype=np.int32, fillvalue=index + 1)
                obs["target_actor_segmentation_id"] = index + 1
        cfg = DatasetConfig(num_points=4, num_actions=3, return_object_pointcloud_only=True, return_rgb=False)
        dataset = GraspDataset(replace(cfg, cache_in_memory=cache), str(h5), str(scenes))
        indices = GraspDataset(cfg, str(h5), str(scenes), indices_only=True)
        device = indices.to_device("cpu")
        assert device["points"].shape == (2, 3, 4, 3)  # shared observations are stored once
        assert device["render"].tolist() == [0, 0, 1]
        for index in range(len(dataset)):
            for seed in range(4):
                np.random.seed(seed)
                expected = dataset[index]
                after = np.random.random(8)
                np.random.seed(seed)
                choices = indices[index]
                np.testing.assert_array_equal(np.random.random(8), after)
                render, draw = device["render"][index], choices["draw"]
                points = device["points"][render, draw].float() + device["center"][render, draw]
                torch.testing.assert_close(points, expected["observations"]["pointcloud"], atol=0, rtol=0)
                torch.testing.assert_close(device["flat_actions"][choices["rows"]], expected["actions"], atol=0, rtol=0)
                assert (expected["observations"]["objectness"] == device["target_id"][render]).all()
        dataset.h5_file.close()
        indices.h5_file.close()

    def test_observations_only_and_splits(self, tmp_path: Path):
        h5, scenes = tmp_path / "data.h5", tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack(
            {split: {split: SceneDescription(scene_id=split)} for split in ("train", "test")}, str(scenes)
        )
        with h5py.File(h5, "w") as output:
            for split in ("train", "test"):
                obs = output.create_group(f"data/{split}/obs")
                obs["pointcloud"] = np.zeros((1, 4, 3), np.float32)
                obs["pointcloud_segmentation"] = np.ones((1, 4, 1), np.int32)
                obs["target_actor_segmentation_id"] = 1
            output["data/train/actions"] = np.zeros((2, 3, 8), np.float32)
        for cache in (False, True):
            cfg = DatasetConfig(num_points=4, return_actions=False, cache_in_memory=cache)
            for split, ids in (("train", ["train"]), ("test", ["test"]), (None, ["test", "train"])):
                dataset = GraspDataset(cfg, str(h5), str(scenes), split=split)
                assert dataset.scene_ids == ids and "actions" not in dataset[0]
                dataset.h5_file.close()

    def test_graspness_sampling_survives_caching(self, tmp_path: Path):
        h5, scenes = tmp_path / "graspness.h5", tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack({"a": SceneDescription(scene_id="a")}, str(scenes))
        with h5py.File(h5, "w") as output:
            group = output.create_group("data/a")
            group["actions"] = np.arange(4 * 3 * 8, dtype=np.float32).reshape(4, 3, 8)
            obs = group.create_group("obs")
            obs["pointcloud"] = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
            obs["pointcloud_segmentation"] = (np.arange(8) % 2).reshape(1, 8, 1)
            obs["target_actor_segmentation_id"] = 1
            obs["graspness_weights"] = np.arange(4, dtype=np.float32)
            obs["best_point_indices"] = np.arange(4)
            obs["action_indices"] = np.arange(4)
        cfg = DatasetConfig(num_points=6, num_actions=2, return_graspness=True, return_object_pointcloud_only=True)
        raw = GraspDataset(cfg, str(h5), str(scenes))
        cached = GraspDataset(replace(cfg, cache_in_memory=True), str(h5), str(scenes))
        for seed in range(4):
            np.random.seed(seed)
            expected = raw[0]
            np.random.seed(seed)
            actual = cached[0]
            for key in expected["observations"]:
                torch.testing.assert_close(actual["observations"][key], expected["observations"][key], atol=0, rtol=0)
            assert actual["observations"]["best_point_indices"].tolist() == [0, 1]
        raw.h5_file.close()
        cached.h5_file.close()

    @pytest.mark.parametrize("cache", [False, True])
    def test_rgb_matches_the_selected_pointcloud_view(self, tmp_path: Path, cache: bool):
        h5, scenes = tmp_path / "rgb.h5", tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack({"a": SceneDescription(scene_id="a")}, str(scenes))
        with h5py.File(h5, "w") as output:
            obs = output.create_group("data/a/obs")
            obs["pointcloud"] = np.broadcast_to(np.arange(3, dtype=np.float32)[:, None, None], (3, 4, 3))
            obs["pointcloud_segmentation"] = np.ones((3, 4, 1), np.int32)
            obs["target_actor_segmentation_id"] = 1
            obs["rgb"] = np.broadcast_to(np.arange(6, dtype=np.uint8)[:, None, None, None], (6, 2, 2, 3))
            obs["qpos"] = np.arange(14, dtype=np.float32).reshape(2, 7)
            obs["camera_poses"] = np.arange(42, dtype=np.float32).reshape(6, 1, 7)
        dataset = GraspDataset(
            DatasetConfig(num_points=2, return_actions=False, cache_in_memory=cache), str(h5), str(scenes)
        )
        for seed in range(4):
            np.random.seed(seed)
            obs = dataset[0]["observations"]
            view = int(obs["pointcloud"][0, 0])
            expected = torch.arange(view * 2, view * 2 + 2).repeat_interleave(3)
            assert (obs["rgb"] == expected[:, None, None]).all()
            torch.testing.assert_close(obs["qpos"], torch.arange(14).reshape(2, 7).float())
            torch.testing.assert_close(obs["camera_poses"], torch.arange(42).reshape(6, 1, 7).float())
        dataset.h5_file.close()
