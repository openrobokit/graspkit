"""Tests for `graspkit.scene.scene_description`."""

import numpy as np
import trimesh

from graspkit.scene.scene_description import BoxDescription, MeshDescription, SceneDescription


class TestSceneDescription:
    def test_msgpack_round_trip(self, tmp_path):
        """save -> load preserves boxes, meshes, and ids; flat dict comes back merged."""
        desc = SceneDescription(
            scene_id="s0",
            target_actor_id="object",
            boxes={"table": BoxDescription("table", (0.5, 0.5, 0.05), (0.0, 0.0, -0.1))},
            meshes={"object": MeshDescription("object", "obj.obj", (1.0, 2.0, 3.0), (0.1, 0.0, 0.0))},
        )
        path = str(tmp_path / "scenes.msgpack")
        SceneDescription.save_scene_descriptions_msgpack({"s0": desc}, path)
        loaded = SceneDescription.load_scene_descriptions_msgpack(path)
        assert set(loaded) == {"s0"}
        got = loaded["s0"]
        assert got.target_actor_id == "object"
        assert got.boxes["table"].size == (0.5, 0.5, 0.05)
        assert got.meshes["object"].collision_mesh_path == "obj.obj"
        assert got.meshes["object"].scales == (1.0, 2.0, 3.0)

    def test_msgpack_return_splits(self, tmp_path):
        """A flat dict is saved under train; return_splits exposes the train/test wrapper."""
        desc = SceneDescription(scene_id="s0", target_actor_id="object")
        path = str(tmp_path / "scenes.msgpack")
        SceneDescription.save_scene_descriptions_msgpack({"s0": desc}, path)
        splits = SceneDescription.load_scene_descriptions_msgpack(path, return_splits=True)
        assert set(splits) == {"train", "test"}
        assert set(splits["train"]) == {"s0"} and splits["test"] == {}

    def test_get_visual_mesh_path(self):
        assert MeshDescription("o", "c.obj", (1.0, 1.0, 1.0)).get_visual_mesh_path() == "c.obj"
        assert (
            MeshDescription("o", "c.obj", (1.0, 1.0, 1.0), visual_mesh_path="v.obj").get_visual_mesh_path() == "v.obj"
        )

    def test_get_bounds_scale_and_translation(self, tmp_path):
        """Bounds = scaled half-extents shifted by position; cube is exact."""
        cube = str(tmp_path / "cube.obj")
        trimesh.creation.box((0.4, 0.4, 0.4)).export(cube)  # half-extent 0.2
        mesh = MeshDescription("o", cube, scales=(2.0, 1.0, 1.0), position=(0.5, 0.0, 0.0))
        bounds = mesh.get_bounds()
        # x: half 0.2 * 2 = 0.4, shifted +0.5 -> (0.1, 0.9); y,z: half 0.2 -> (-0.2, 0.2)
        np.testing.assert_allclose(bounds[0], (0.1, 0.9), atol=1e-6)
        np.testing.assert_allclose(bounds[1], (-0.2, 0.2), atol=1e-6)
        np.testing.assert_allclose(bounds[2], (-0.2, 0.2), atol=1e-6)

    def test_get_bounds_uses_cached_field(self):
        """A pre-cached `bounds` is returned without touching the mesh file."""
        cached = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
        mesh = MeshDescription("o", "does_not_exist.obj", (1.0, 1.0, 1.0), bounds=cached)
        assert mesh.get_bounds() == cached
