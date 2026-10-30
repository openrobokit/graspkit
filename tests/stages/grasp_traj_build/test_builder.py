"""Tests for `graspkit.stages.grasp_traj_build.builder`."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp

from graspkit.configs.experiment_config import GraspTrajBuilderConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_traj_build.builder import GraspTrajBuilder


class TestGraspTrajBuilder:
    @pytest.mark.parametrize("plan_table_only", [False, True])
    def test_builder_preserves_each_table_and_restores_scene_identity(self, plan_table_only: bool):
        tables = [
            BoxDescription("table", (1.0, 0.5, 0.2), (0.0, 0.0, -0.1)),
            BoxDescription("table", (1.6, 0.8, 0.7), (1.0, -1.0, 0.4), (2**-0.5, 0.0, 0.0, 2**-0.5)),
        ]
        scenes = [
            SceneDescription(
                scene_id=f"scene-{index}",
                target_actor_id="object",
                boxes={"table": table, "wall": BoxDescription("wall", (0.1, 1.0, 1.0))},
                meshes={"object": MeshDescription("object", "unused.obj", (1.0, 1.0, 1.0))},
            )
            for index, table in enumerate(tables)
        ]
        qpos = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
        batch = GraspKitData(
            scene_ids=[scene.scene_id for scene in scenes],
            scene_descriptions=scenes,
            pre_grasp_qpos=qpos - 1,
            grasp_qpos=qpos,
            post_grasp_qpos=qpos + 1,
        )
        planner = Mock()
        planner.plan.side_effect = lambda data, **kwargs: replace(data, full_traj_qpos=qpos[:, :, None])
        builder = GraspTrajBuilder.__new__(GraspTrajBuilder)
        builder.cfg = GraspTrajBuilderConfig(plan_table_only=plan_table_only)
        builder.robot = SimpleNamespace(device="cpu")
        builder.motion_planner = planner
        start = qpos + 2
        planned, success = builder.build(batch, start_qpos=start, skip_filter=True)
        planner.plan.assert_called_once()
        data = planner.plan.call_args.args[0]
        assert planner.plan.call_args.kwargs == {"start_qpos": start, "return_pre_post_only": False}
        assert data.pre_grasp_qpos is batch.pre_grasp_qpos
        assert data.grasp_qpos is batch.grasp_qpos
        assert data.post_grasp_qpos is batch.post_grasp_qpos
        assert [scene.boxes["table"] for scene in data.scene_descriptions] == tables
        if plan_table_only:
            assert data.scene_ids == ["table_only", "table_only"]
            assert all(scene.target_actor_id == "table" for scene in data.scene_descriptions)
            assert all(not scene.meshes and set(scene.boxes) == {"table"} for scene in data.scene_descriptions)
            geometry = SceneGeom(data.scene_descriptions, device="cpu")
            points = wp.from_numpy(
                np.asarray([[0.0, 0.0, 0.05], [1.0, -1.0, 0.8]], np.float32), dtype=wp.vec3, device="cpu"
            )
            indices = wp.from_numpy(np.arange(2, dtype=np.int32), dtype=wp.int32, device="cpu")
            distances = geometry.collision_scene.query_sdf(points, scene_indices=indices, distance_only=True)
            np.testing.assert_allclose(distances.numpy(), [0.05, 0.05], atol=1e-6)
        else:
            assert data.scene_descriptions is scenes and data.scene_ids == batch.scene_ids
        assert planned.scene_descriptions is scenes and planned.scene_ids == batch.scene_ids
        assert batch.scene_descriptions is scenes and batch.scene_ids == ["scene-0", "scene-1"]
        assert success.tolist() == [[True], [True]]
