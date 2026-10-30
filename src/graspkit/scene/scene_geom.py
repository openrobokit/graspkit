"""Thin wrapper turning `List[SceneDescription]` into runtime `WarpScene`s; BVH/SDF baking lives in `robokit.geom`."""

from typing import List, Literal, Optional, Tuple

import msgspec
import numpy as np
import torch
import trimesh
from robokit.geom import SdfVolume, WarpScene
from robokit.xform.numpy import inverse_tf_mat, matrix_to_quaternion, quaternion_to_matrix
from torch.types import Device

from graspkit.scene.scene_description import SceneDescription
from graspkit.scene.warp_scene_builder import MeshSpec, PrimitiveSpec, SceneSpec, SdfSpec, build_warp_scene


class SceneGeom:
    """Per-scene mesh-backed WarpScenes for grasp / motion optimizers."""

    def __init__(self, scene_descriptions: List[SceneDescription], device: Device = "cpu"):
        d = torch.device(device)
        self.scene_descriptions = scene_descriptions
        self.device = device
        self._device_str = f"cuda:{d.index or 0}" if d.type == "cuda" else "cpu"
        self._target_scene: Optional[WarpScene] = None
        self._solver_scenes: Optional[Tuple[WarpScene, WarpScene]] = None
        self._solver_settings: Optional[tuple] = None
        self._scene_key = msgspec.msgpack.encode(scene_descriptions)
        self.collision_scene = build_warp_scene(
            [self._collision_spec(s) for s in scene_descriptions], device=self._device_str
        )

    @property
    def target_scene(self) -> WarpScene:
        """The target mesh of each scene (none for a table-only scene); built lazily on first access."""
        if self._target_scene is None:
            specs = [
                SceneSpec(meshes={"target": self._collision_spec(s).meshes[s.target_actor_id]} if s.meshes else {})
                for s in self.scene_descriptions
            ]
            self._target_scene = build_warp_scene(specs, device=self._device_str)
        return self._target_scene

    def __len__(self) -> int:
        return len(self.scene_descriptions)

    def update(self, scene_descriptions: List[SceneDescription]) -> None:
        """Update geometry in place for a scene batch with the same layout; captured pointers stay valid."""
        key = msgspec.msgpack.encode(scene_descriptions)
        if key == self._scene_key:
            return
        self.scene_descriptions = scene_descriptions
        self._scene_key = key
        build_warp_scene([self._collision_spec(s) for s in scene_descriptions], self._device_str, self.collision_scene)
        if self._target_scene is not None:
            build_warp_scene(
                [
                    SceneSpec(meshes={"target": self._collision_spec(s).meshes[s.target_actor_id]} if s.meshes else {})
                    for s in scene_descriptions
                ],
                self._device_str,
                self._target_scene,
            )
        if self._solver_scenes is not None:
            self.solver_scenes("sdf_volume", *self._solver_settings[0])

    def _collision_spec(self, desc: SceneDescription, T_robot_world: Optional[np.ndarray] = None) -> SceneSpec:
        """Every mesh + box of ``desc``, re-expressed in the robot base frame when ``T_robot_world`` is given."""

        def pose(position: Tuple[float, ...], quaternion: Tuple[float, ...]) -> Tuple[tuple, tuple]:
            if T_robot_world is None:
                return position, quaternion
            rot = T_robot_world[:3, :3] @ quaternion_to_matrix(np.asarray(quaternion, dtype=np.float64))
            trans = T_robot_world[:3, :3] @ np.asarray(position, dtype=np.float64) + T_robot_world[:3, 3]
            return tuple(trans.tolist()), tuple(matrix_to_quaternion(rot).tolist())

        return SceneSpec(
            meshes={
                k: MeshSpec(m.collision_mesh_path, m.scales, *pose(m.position, m.quaternion))
                for k, m in desc.meshes.items()
            },
            primitives={
                k: PrimitiveSpec("box", b.size, *pose(b.position, b.quaternion)) for k, b in desc.boxes.items()
            },
        )

    def build_collision_scene_in_base(self, T_world_robot: np.ndarray) -> WarpScene:
        """One ``WarpScene`` (mesh+box) for ``MotionPlanner.plan()``, in robot base frame (FK is base-relative)."""
        T_robot_world = inverse_tf_mat(np.asarray(T_world_robot, dtype=np.float64))
        return build_warp_scene(
            [self._collision_spec(d, T_robot_world) for d in self.scene_descriptions], device=self._device_str
        )

    def query_target_object_signed_distances(
        self, query_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable per-scene target-object SDF (single mesh -> diff path)."""
        return self._query(self.target_scene, query_points)

    def query_scene_signed_distances(
        self, query_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-scene SDF (meshes+primitives), differentiable; drives penetration gradient in motion/grasp-GD."""
        return self._query(self.collision_scene, query_points)

    def _query(self, scene: WarpScene, query_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qp = query_points.contiguous()
        flat = qp.reshape(-1, 3)
        offsets = torch.arange(0, flat.shape[0] + 1, qp.shape[1], device=flat.device)
        sdf, normals, clst = scene.query_sdf_torch(flat, offsets)
        return sdf.reshape_as(qp[..., 0]), normals.reshape_as(qp), clst.reshape_as(qp)

    def solver_scenes(
        self,
        backend: Literal["mesh", "sdf_volume"],
        sdf_bake_padding: float = 0.25,
        sdf_resolution: int = 64,
        sdf_voxel_size: Optional[float] = None,
    ) -> Tuple[WarpScene, WarpScene]:
        """`(target_scene, collision_scene)`; `sdf_volume` swaps the target mesh for an SDF (params in mesh units)."""
        if backend == "mesh":
            return self.target_scene, self.collision_scene
        if backend != "sdf_volume":
            raise ValueError(f"Unsupported scene_geom_backend: {backend!r}")
        settings = (sdf_bake_padding, sdf_resolution, sdf_voxel_size)
        if self._solver_scenes is not None and self._solver_settings == (settings, self._scene_key):
            return self._solver_scenes
        target_specs, collision_specs = [], []
        for desc in self.scene_descriptions:
            sdfs = {}
            if desc.meshes:  # a table-only scene collides against its boxes alone
                assert len(desc.meshes) == 1, f"scene {desc.scene_id}: sdf_volume needs the target as the only mesh"
                m = desc.meshes[desc.target_actor_id]
                voxel = sdf_voxel_size
                if voxel is None:
                    bounds = trimesh.load(m.collision_mesh_path, force="mesh").bounds
                    voxel = float((bounds[1] - bounds[0] + 2.0 * sdf_bake_padding).max()) / sdf_resolution
                volume = SdfVolume.from_mesh(
                    m.collision_mesh_path, voxel, sdf_bake_padding, device=self._device_str, cache=True
                )
                sdfs = {"target": SdfSpec(volume, m.scales, m.position, m.quaternion)}
            target_specs.append(SceneSpec(sdfs=sdfs))
            collision_specs.append(SceneSpec(sdfs=sdfs, primitives=self._collision_spec(desc).primitives))
        previous = self._solver_scenes or (None, None)
        self._solver_scenes = (
            build_warp_scene(target_specs, self._device_str, previous[0]),
            build_warp_scene(collision_specs, self._device_str, previous[1]),
        )
        self._solver_settings = (settings, self._scene_key)
        return self._solver_scenes
