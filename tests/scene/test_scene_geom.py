"""Tests for `graspkit.scene.scene_geom`."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import trimesh
import warp as wp

from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.scene.warp_scene_builder import MeshSpec, PrimitiveSpec, SceneSpec, build_warp_scene


class TestSceneGeom:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    @staticmethod
    def _cube_scene(tmp_path: Path, with_box: bool = True) -> SceneDescription:
        cube = str(tmp_path / "cube.obj")
        trimesh.creation.box((0.4, 0.4, 0.4)).export(cube)  # half-extent 0.2
        boxes = {"table": BoxDescription("table", (0.5, 0.5, 0.05), (0.0, 0.0, -0.5))} if with_box else {}
        return SceneDescription(
            scene_id="s0",
            target_actor_id="object",
            boxes=boxes,
            meshes={"object": MeshDescription("object", cube, (1.0, 1.0, 1.0))},
        )

    def test_target_object_sdf_matches_cube(self, tmp_path):
        """Differentiable target SDF reproduces the analytical cube SDF (half-extent 0.2)."""
        geom = SceneGeom([self._cube_scene(tmp_path, with_box=False)], device="cpu")
        qp = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.5]]], dtype=torch.float32)
        sdf, _, _ = geom.query_target_object_signed_distances(qp)
        torch.testing.assert_close(sdf, torch.tensor([[0.3, -0.2, 0.1, 0.3]]), atol=1e-5, rtol=1e-5)

    def test_collision_sdf_matches_direct_warp_scene(self, tmp_path):
        """Adapter parity: SceneGeom collision matches a hand-built WarpScene with same MeshSpec + box PrimitiveSpec."""
        desc = self._cube_scene(tmp_path, with_box=True)
        geom = SceneGeom([desc], device="cpu")
        direct = build_warp_scene(
            [
                SceneSpec(
                    meshes={"object": MeshSpec(mesh_path=desc.meshes["object"].collision_mesh_path)},
                    primitives={"table": PrimitiveSpec(type="box", params=(0.5, 0.5, 0.05), position=(0.0, 0.0, -0.5))},
                )
            ],
            device="cpu",
        )
        qp = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, -0.45]]], dtype=torch.float32)
        sdf_geom, _, _ = geom.query_scene_signed_distances(qp)
        sdf_direct, _, _ = direct.query_sdf_torch(qp.reshape(-1, 3), torch.tensor([0, qp.shape[0] * qp.shape[1]]))
        torch.testing.assert_close(sdf_geom, sdf_direct.reshape(1, -1), atol=1e-5, rtol=1e-5)

    def test_solver_scenes_mesh_returns_built_scenes(self, tmp_path):
        geom = SceneGeom([self._cube_scene(tmp_path)], device="cpu")
        target, collision = geom.solver_scenes(backend="mesh")
        assert target is geom.target_scene and collision is geom.collision_scene

    def test_update_keeps_scene_buffers_and_matches_fresh_geometry(self, tmp_path: Path):
        scenes = [self._cube_scene(tmp_path)]
        geom = SceneGeom(scenes, device="cpu")
        target, collision = geom.solver_scenes("mesh")
        pointers = [(g.mesh_ids_wp.ptr, g.inv_poses_wp.ptr, g.scales_wp.ptr) for g in target.geoms]
        box = collision.geoms[-1]
        box_ptr = box.params_wp.ptr
        scenes[0].meshes["object"] = replace(
            scenes[0].meshes["object"], scales=(0.5, 0.5, 0.5), position=(0.3, 0.0, 0.0)
        )
        scenes[0].boxes["table"] = replace(scenes[0].boxes["table"], size=(0.1, 0.1, 0.2))
        geom.update(scenes)
        assert geom.solver_scenes("mesh") == (target, collision)
        assert pointers == [(g.mesh_ids_wp.ptr, g.inv_poses_wp.ptr, g.scales_wp.ptr) for g in target.geoms]
        assert box.params_wp.ptr == box_ptr
        qp = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, -0.45]]])
        fresh = SceneGeom(scenes, device="cpu")
        for actual, expected in zip(geom.query_scene_signed_distances(qp), fresh.query_scene_signed_distances(qp)):
            torch.testing.assert_close(actual, expected)

    def test_solver_scenes_rejects_unknown_backend(self, tmp_path):
        geom = SceneGeom([self._cube_scene(tmp_path)], device="cpu")
        with pytest.raises(ValueError):
            geom.solver_scenes(backend="nonsense")

    @pytest.mark.cuda
    def test_solver_scenes_sdf_volume_matches_mesh(self, tmp_path):
        """sdf_volume target SDF agrees with the mesh backend within a baking tolerance."""
        geom = SceneGeom([self._cube_scene(tmp_path, with_box=False)], device="cuda:0")
        target_sdf_scene, _ = geom.solver_scenes(backend="sdf_volume", sdf_resolution=64, sdf_bake_padding=0.2)
        qp = torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.1, 0.0]]], dtype=torch.float32, device="cuda:0")
        flat, first = qp.reshape(-1, 3), torch.tensor([0, qp.shape[0] * qp.shape[1]], device="cuda:0")
        sdf_vol, _, _ = target_sdf_scene.query_sdf_torch(flat, first)
        sdf_mesh, _, _ = geom.target_scene.query_sdf_torch(flat, first)
        torch.testing.assert_close(sdf_vol, sdf_mesh, atol=0.02, rtol=0.0)
