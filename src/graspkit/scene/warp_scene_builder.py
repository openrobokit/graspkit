"""Per-primitive scene specs (grouped per scene by `SceneSpec`) + `build_warp_scene`, which builds one `WarpScene`.

One spec type per geom kind, mirroring robokit's typed geoms:

- :class:`MeshSpec` -- triangle mesh loaded from a file path (BVH-backed; optional baked-SDF far tier).
- :class:`SdfSpec` -- pre-baked SDF volume.
- :class:`PrimitiveSpec` -- analytical primitive (box).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import warp as wp
from robokit.geom.geoms import BoxGeom, MeshGeom, VolumeGeom
from robokit.geom.scene import WarpScene
from robokit.geom.sdf_volume import SdfVolume
from robokit.lie.se3 import se3_to_matrix
from robokit.utils.warp_utils import wp_vec7


def _poses_wp(
    positions: List[Tuple[float, float, float]], quats: List[Tuple[float, float, float, float]], device: str
) -> wp.array:
    xyz_wxyz = np.concatenate([np.asarray(positions, dtype=np.float32), np.asarray(quats, dtype=np.float32)], axis=1)
    return se3_to_matrix(wp.from_numpy(xyz_wxyz, dtype=wp_vec7, device=device))


@dataclass
class MeshSpec:
    """Triangle mesh from a file (BVH-backed SDF); ``enable_sdf`` adds a disk-cached baked far tier near the surface."""

    mesh_path: str = ""
    scales: Union[float, Tuple[float, float, float]] = 1.0  # warp consumes scales[0] (isotropic)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    enable_sdf: bool = False
    sdf_voxel_size: float = 0.01
    sdf_padding: float = 0.1
    sdf_refine_band_voxels: Optional[float] = None  # None -> robokit default band


@dataclass
class SdfSpec:
    """Pre-baked SDF volume."""

    volume: SdfVolume
    scales: Union[float, Tuple[float, float, float]] = 1.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class PrimitiveSpec:
    """Analytical primitive (box).

    ``params`` packs type-specific shape values in the local frame:

    - ``"box"``:     ``(x, y, z)`` full extents.
    """

    type: Literal["box"] = "box"
    params: Tuple[float, ...] = ()
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneSpec:
    """Per-scene geometry: three dict-keyed groups by primitive kind."""

    meshes: Dict[str, MeshSpec] = field(default_factory=dict)
    sdfs: Dict[str, SdfSpec] = field(default_factory=dict)
    primitives: Dict[str, PrimitiveSpec] = field(default_factory=dict)


def build_warp_scene(specs: List[SceneSpec], device: str = "cpu", scene: Optional[WarpScene] = None) -> WarpScene:
    """Build a WarpScene, or update an existing scene with the same geometry layout in place."""
    device = str(device)
    n = len(specs)
    updating = scene is not None
    if scene is None:
        scene = WarpScene(num_scenes=n, device=device)
    assert scene.num_scenes == n, "updating a scene requires the same number of scenes"
    geom_index = 0

    def scalar(scales) -> float:
        return float(scales if isinstance(scales, (int, float)) else scales[0])

    # Meshes with the SDF far tier need a different kernel, so they form their own MeshGeom group.
    for want_sdf in (False, True):
        paths, positions, quats, scales, offsets = [], [], [], [], [0]
        sdf_settings = None
        for spec in specs:
            for m in spec.meshes.values():
                if m.enable_sdf != want_sdf:
                    continue
                paths.append(m.mesh_path)
                positions.append(m.position)
                quats.append(m.quaternion)
                scales.append(scalar(m.scales))
                if want_sdf:
                    settings = (m.sdf_voxel_size, m.sdf_padding, m.sdf_refine_band_voxels)
                    assert sdf_settings in (None, settings), "one SDF-tier setting per scene list"
                    sdf_settings = settings
            offsets.append(len(paths))
        if paths:
            poses = _poses_wp(positions, quats, device)
            scales_wp = wp.from_numpy(np.asarray(scales, dtype=np.float32), dtype=wp.float32, device=device)
            if updating:
                geom = scene.geoms[geom_index]
                assert isinstance(geom, MeshGeom) and geom.enable_sdf == want_sdf, "geometry layout changed"
                geom.update(meshes=paths, scene_offsets=np.asarray(offsets, np.int32), poses=poses, scales=scales_wp)
                geom_index += 1
                continue
            kwargs = {}
            if want_sdf:
                voxel, padding, band = sdf_settings
                kwargs = {"enable_sdf": True, "sdf_voxel_size": voxel, "sdf_padding": padding}
                if band is not None:
                    kwargs["refine_band_voxels"] = band
            scene = scene.add(
                MeshGeom(
                    paths,
                    np.asarray(offsets, dtype=np.int32),
                    poses=poses,
                    scales=scales_wp,
                    **kwargs,
                )
            )

    volumes, positions, quats, scales, offsets = [], [], [], [], [0]
    for spec in specs:
        for s in spec.sdfs.values():
            volumes.append(s.volume)
            positions.append(s.position)
            quats.append(s.quaternion)
            scales.append(scalar(s.scales))
        offsets.append(len(volumes))
    if volumes:
        poses = _poses_wp(positions, quats, device)
        scales_wp = wp.from_numpy(np.asarray(scales, dtype=np.float32), dtype=wp.float32, device=device)
        if updating:
            geom = scene.geoms[geom_index]
            assert isinstance(geom, VolumeGeom), "geometry layout changed"
            geom.update(sdf_volumes=volumes, scene_offsets=np.asarray(offsets, np.int32), poses=poses, scales=scales_wp)
            geom_index += 1
        else:
            scene = scene.add(VolumeGeom(volumes, np.asarray(offsets, np.int32), poses=poses, scales=scales_wp))

    half_extents, positions, quats, offsets = [], [], [], [0]
    for spec in specs:
        for p in spec.primitives.values():
            assert p.type == "box", f"unsupported primitive type: {p.type}"
            half_extents.append(0.5 * np.asarray(p.params, dtype=np.float32))
            positions.append(p.position)
            quats.append(p.quaternion)
        offsets.append(len(half_extents))
    if half_extents:
        poses = _poses_wp(positions, quats, device)
        if updating:
            geom = scene.geoms[geom_index]
            assert isinstance(geom, BoxGeom), "geometry layout changed"
            geom.update(half_extents=np.stack(half_extents), scene_offsets=np.asarray(offsets, np.int32), poses=poses)
            geom_index += 1
        else:
            scene = scene.add(BoxGeom(np.stack(half_extents), np.asarray(offsets, np.int32), poses=poses))
    assert not updating or geom_index == len(scene.geoms), "geometry layout changed"
    return scene
