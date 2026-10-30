"""Tests for `graspkit.configs.experiment_config`."""

from graspkit.configs.experiment_config import DatasetConfig


class TestDatasetConfig:
    def test_dataset_positional_arguments_are_unchanged(self):
        cfg = DatasetConfig(8, 32, True, False, True)
        assert cfg.num_actions == 8 and cfg.num_points == 32
        assert cfg.return_actions and cfg.return_object_pointcloud_only
        assert not cfg.return_graspness and not cfg.cache_in_memory
