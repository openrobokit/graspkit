"""Tests for `graspkit.mani_skill.utils.scene_builder.table.scene_builder`."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("mani_skill")  # the maniskill extra

import sapien
import torch

from graspkit.mani_skill.utils.scene_builder.table import scene_builder


class TestTableSceneBuilder:
    @pytest.mark.parametrize("can_render", [False, True])
    def test_table_build_preserves_collision_and_pose_without_render_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, can_render: bool
    ):
        table = SimpleNamespace(set_pose=Mock())
        actor_builder = Mock(return_value=table)
        actor_builder.build_kinematic.return_value = table
        scene = SimpleNamespace(can_render=can_render, create_actor_builder=Mock(return_value=actor_builder))
        monkeypatch.setattr(scene_builder, "_get_maniskill_table_assets_dir", lambda: tmp_path)
        builder = scene_builder.TableSceneBuilder(SimpleNamespace(scene=scene))
        box_size = (1.6, 0.8, 0.6)
        builder.build([box_size])
        scene.create_actor_builder.assert_called_once_with()
        actor_builder.add_box_collision.assert_called_once()
        collision = actor_builder.add_box_collision.call_args.kwargs
        assert collision["half_size"] == (0.8, 0.4, 0.3)
        np.testing.assert_allclose(collision["pose"].p, (0.0, 0.0, 0.3))
        np.testing.assert_allclose(collision["pose"].q, (1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(actor_builder.initial_pose.p, (0.0, 0.0, -0.6))
        np.testing.assert_allclose(actor_builder.initial_pose.q, (1.0, 0.0, 0.0, 0.0))
        actor_builder.build_kinematic.assert_called_once_with(name="table-workspace")
        assert builder.table is table and builder.scene_objects == [table]
        pose = sapien.Pose(p=(1.0, 2.0, 3.0))
        builder.initialize(torch.tensor([0]), pose)
        table.set_pose.assert_called_once_with(pose)
