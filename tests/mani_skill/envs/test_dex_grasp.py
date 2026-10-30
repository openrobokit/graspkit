"""Tests for `graspkit.mani_skill.envs.dex_grasp`."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mani_skill")  # the maniskill extra

from graspkit.configs.experiment_config import EnvConfig
from graspkit.mani_skill.envs import dex_grasp


class TestMsDexGraspEnv:
    def test_mesh_actor_creation_order(self, monkeypatch: pytest.MonkeyPatch):
        helper = MagicMock()
        helper.mesh_keys = [("b.glb", (1, 1, 1)), ("a.glb", (1, 1, 1)), ("b.glb", (1, 1, 1))]
        helper.get_scene_indices_by_mesh_key.return_value = [0]
        helper.get_actor_ids_by_mesh_key.return_value = ["target"]
        helper.get_positions_by_actor_id.return_value = [(0, 0, 0)]
        helper.get_quaternions_by_actor_id.return_value = [(1, 0, 0, 0)]
        monkeypatch.setattr(dex_grasp, "SceneDescriptionsHelper", MagicMock(return_value=helper))
        builder = MagicMock()
        builder.set_scene.return_value = builder
        builder.build.return_value._objs = [MagicMock()]
        monkeypatch.setattr(dex_grasp, "ActorBuilder", MagicMock(return_value=builder))
        monkeypatch.setattr(dex_grasp.Actor, "create_from_entities", MagicMock(return_value=MagicMock()))
        env = SimpleNamespace(config=EnvConfig(floating_object=True), scene=MagicMock())
        dex_grasp.MsDexGraspEnv._load_scene(env, {"scene_descriptions": [SimpleNamespace(target_actor_id="target")]})
        assert [c.args[0] for c in builder.add_multiple_convex_collisions_from_file.call_args_list] == [
            "a.glb",
            "b.glb",
        ]
