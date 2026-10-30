"""Inspect mesh data from different sources: per-category stats + ManiSkill sim comparison.

CUDA_VISIBLE_DEVICES=0 uv run scripts/stages/asset_prep/inspect_data.py
"""

import random
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien.physx as physx
import trimesh
import viser
from mani_skill.utils.geometry.trimesh_utils import merge_meshes
from sapien.render import RenderBodyComponent

from graspkit.configs.experiment_config import EnvConfig
from graspkit.constants import PROJECT_ROOT
from graspkit.mani_skill.agents.xhand import XHandRightFloatingBase  # noqa: F401
from graspkit.mani_skill.envs.dex_grasp import MsDexGraspEnv
from graspkit.scene.scene_description import MeshDescription, SceneDescription

VISER_CONFIG: dict[str, tuple[float, tuple[float, float, float], str]] = {
    "disk_collision": (-2.0, (0.2, 0.6, 0.9), "Disk Collision"),
    "disk_visual": (-0.67, (0.9, 0.4, 0.1), "Disk Visual"),
    "sapien_collision": (0.67, (0.2, 0.9, 0.6), "SAPIEN Collision"),
    "sapien_visual": (2.0, (0.9, 0.8, 0.2), "SAPIEN Visual"),
}

DATA_DIR = PROJECT_ROOT / "data" / "object_data"
OUTPUT_DIR = Path("/tmp/inspect_data_output")
SAMPLES_PER_SOURCE = 5


def discover_samples(
    n: int = SAMPLES_PER_SOURCE,
) -> list[tuple[str, Path, Path | None]]:
    """Sample n mesh files from each data source. Returns (source, collision_path, visual_path)."""
    random.seed(42)
    # Each source: (collision_glob, visual_path_fn), mapping a collision path to a visual path or None.
    sources: dict[str, tuple[str, ...]] = {
        "dex_diffuser/ycb": sorted((DATA_DIR / "dex_diffuser" / "processed" / "ycb").glob("*/collision.glb")),
        "dex_diffuser/egad": sorted((DATA_DIR / "dex_diffuser" / "processed" / "egad").glob("*/collision.glb")),
        "dex_diffuser/kit": sorted((DATA_DIR / "dex_diffuser" / "processed" / "kit").glob("*/collision.glb")),
        "dex_diffuser/contactdb": sorted(
            (DATA_DIR / "dex_diffuser" / "processed" / "contactdb").glob("*/collision.glb")
        ),
        "dexgraspnet": sorted((DATA_DIR / "dexgraspnet").glob("*/coacd/decomposed.obj")),
        "objaverse_dex1b": sorted((DATA_DIR / "objaverse_dex1b" / "processed").glob("*/collision.glb")),
        "robocasa": sorted((DATA_DIR / "robocasa" / "processed").glob("*/*/collision.glb")),
    }
    # Sources with a visual.glb sibling next to the collision mesh
    sources_with_visual = {
        "dex_diffuser/ycb",
        "dex_diffuser/egad",
        "dex_diffuser/kit",
        "dex_diffuser/contactdb",
        "robocasa",
    }

    samples = []
    for source_name, files in sources.items():
        if not files:
            print(f"  SKIP {source_name}: no files found")
            continue
        picked = random.sample(files, min(n, len(files)))
        for f in picked:
            visual = f.parent / "visual.glb" if source_name in sources_with_visual else None
            samples.append((source_name, f, visual))
    print(f"Discovered {len(samples)} samples from {len(sources)} sources\n")
    return samples


def load_mesh_with_groups(path: Path) -> tuple[trimesh.Trimesh, int]:
    """Returns (merged mesh, number of geometry groups in file)."""
    loaded = trimesh.load(str(path), process=False, split_objects=True)
    if isinstance(loaded, trimesh.Scene):
        parts = list(loaded.geometry.values())
        return merge_meshes(parts), len(parts)
    return loaded, 1


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    centered = mesh.vertices - mesh.bounding_box.centroid
    extent = mesh.bounding_box.extents.max()
    scale = 1.0 / extent if extent > 0 else 1.0
    return trimesh.Trimesh(vertices=centered * scale, faces=mesh.faces)


def get_component_meshes(component: physx.PhysxRigidBaseComponent) -> list[trimesh.Trimesh]:
    """Get collision meshes from a physx component."""
    meshes = []
    for geom in component.get_collision_shapes():
        if isinstance(geom, (physx.PhysxCollisionShapeConvexMesh, physx.PhysxCollisionShapeTriangleMesh)):
            mesh = trimesh.Trimesh(vertices=geom.vertices * geom.scale, faces=geom.get_triangles())
            mesh.apply_transform(geom.get_local_pose().to_transformation_matrix())
            meshes.append(mesh)
    return meshes


def get_visual_mesh(entity) -> trimesh.Trimesh:
    """Extract visual mesh from a SAPIEN entity."""
    render_body: RenderBodyComponent = entity.find_component_by_type(RenderBodyComponent)
    meshes = []
    for render_shape in render_body.render_shapes:
        local_pose = render_shape.get_local_pose().to_transformation_matrix()
        for part in render_shape.parts:
            mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.triangles)
            mesh.apply_transform(local_pose)
            meshes.append(mesh)
    return merge_meshes(meshes)


def main():
    samples = discover_samples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Part 1: Mesh stats (per-source summary) ---
    stats_by_source = defaultdict(list)
    mesh_paths_rel = []

    for source, collision_path, _visual_path in samples:
        mesh, n_parts = load_mesh_with_groups(collision_path)
        size = mesh.bounds[1] - mesh.bounds[0]
        center = (mesh.bounds[1] + mesh.bounds[0]) / 2
        stats_by_source[source].append(
            {
                "faces": len(mesh.faces),
                "size": size,
                "center": center,
                "n_parts": n_parts,
            }
        )
        mesh_paths_rel.append(str(collision_path.relative_to(PROJECT_ROOT)))

    print("=== Mesh Stats (per-source avg) ===")
    print(f"{'source':<25} {'n':>4} {'parts':>6} {'faces':>8}  {'size (x, y, z)':>22}  {'center (x, y, z)':>24}")
    print("-" * 97)
    for source, stats_list in stats_by_source.items():
        n = len(stats_list)
        avg_parts = np.mean([s["n_parts"] for s in stats_list])
        avg_faces = np.mean([s["faces"] for s in stats_list])
        avg_size = np.mean([s["size"] for s in stats_list], axis=0)
        avg_center = np.mean([s["center"] for s in stats_list], axis=0)
        size_str = f"({avg_size[0]:.3f}, {avg_size[1]:.3f}, {avg_size[2]:.3f})"
        center_str = f"({avg_center[0]:.3f}, {avg_center[1]:.3f}, {avg_center[2]:.3f})"
        print(f"{source:<25} {n:>4} {avg_parts:>6.1f} {avg_faces:>8.0f}  {size_str:>22}  {center_str:>24}")

    # --- Part 2: ManiSkill sim comparison (per-source avg) ---
    print("\n=== ManiSkill Sim Comparison (per-source avg) ===")
    env_config = EnvConfig(floating_object=True)
    env: MsDexGraspEnv = gym.make(
        "DexGrasp",
        robot_uids="xhand_right_floating_base",
        num_envs=1,
        config=env_config,
        render_mode="rgb_array",
    ).unwrapped

    sim_stats_by_source = defaultdict(list)
    viewer_data: list[dict] = []

    def _mesh_size(m):
        return m.bounds[1] - m.bounds[0] if m is not None else np.zeros(3)

    def _vf(m):
        return (len(m.vertices), len(m.faces)) if m is not None else (0, 0)

    for (source, collision_path, visual_path), rel_path in zip(samples, mesh_paths_rel):
        obj_name = (
            collision_path.parent.name if collision_path.stem in ("decomposed", "collision") else collision_path.stem
        )

        disk_collision, disk_collision_groups = load_mesh_with_groups(collision_path)
        disk_visual, disk_visual_groups = load_mesh_with_groups(visual_path) if visual_path else (None, 0)

        visual_rel = str(visual_path.relative_to(PROJECT_ROOT)) if visual_path else None
        scene_desc = SceneDescription(
            scene_id="inspect",
            boxes={},
            meshes={
                "object_0": MeshDescription(
                    actor_id="object_0",
                    collision_mesh_path=rel_path,
                    visual_mesh_path=visual_rel,
                    scales=(1.0, 1.0, 1.0),
                    position=(0.0, 0.0, 0.3),
                    quaternion=(1.0, 0.0, 0.0, 0.0),
                )
            },
            target_actor_id="object_0",
        )
        env.reset(options={"scene_descriptions": [scene_desc]})

        entity = env.target_actor._objs[0]
        collision_meshes = []
        for comp in entity.components:
            if isinstance(comp, physx.PhysxRigidBaseComponent):
                collision_meshes.extend(get_component_meshes(comp))
        sapien_collision = merge_meshes(collision_meshes) if collision_meshes else None
        sapien_visual = get_visual_mesh(entity)

        bbox_diff_coll = float(np.max(np.abs(_mesh_size(disk_collision) - _mesh_size(sapien_collision))))
        bbox_diff_vis = (
            float(np.max(np.abs(_mesh_size(disk_visual) - _mesh_size(sapien_visual)))) if disk_visual else 0.0
        )

        # Save meshes
        out_dir = OUTPUT_DIR / source / obj_name
        out_dir.mkdir(parents=True, exist_ok=True)
        disk_collision.export(str(out_dir / "disk_collision.stl"))
        if disk_visual:
            disk_visual.export(str(out_dir / "disk_visual.stl"))
        if sapien_collision:
            sapien_collision.export(str(out_dir / "sapien_collision.stl"))
        if sapien_visual:
            sapien_visual.export(str(out_dir / "sapien_visual.stl"))

        dc_v, dc_f = _vf(disk_collision)
        dv_v, dv_f = _vf(disk_visual)
        sc_v, sc_f = _vf(sapien_collision)
        sv_v, sv_f = _vf(sapien_visual)

        sim_stats_by_source[source].append(
            {
                "dc_verts": dc_v,
                "dc_faces": dc_f,
                "dv_verts": dv_v,
                "dv_faces": dv_f,
                "sc_verts": sc_v,
                "sc_faces": sc_f,
                "sv_verts": sv_v,
                "sv_faces": sv_f,
                "bbox_diff_coll": bbox_diff_coll,
                "bbox_diff_vis": bbox_diff_vis,
            }
        )
        viewer_data.append(
            {
                "source": source,
                "obj_name": obj_name,
                "disk_collision": disk_collision,
                "disk_visual": disk_visual,
                "sapien_collision": sapien_collision,
                "sapien_visual": sapien_visual,
                "groups": {
                    "disk_collision": disk_collision_groups,
                    "disk_visual": disk_visual_groups,
                    "sapien_collision": len(collision_meshes),
                    "sapien_visual": len(render_body.render_shapes)
                    if (render_body := entity.find_component_by_type(RenderBodyComponent))
                    else 0,
                },
            }
        )

    env.close()

    # Header: group v/f per mesh type for clarity
    print(
        f"{'source':<25} {'n':>4}  "
        f"{'disk_coll':>17}  {'disk_vis':>17}  {'sapien_coll':>17}  {'sapien_vis':>17}  "
        f"{'bb_coll':>10} {'bb_vis':>10}"
    )
    print(
        f"{'':25} {'':4}  "
        f"{'v':>8} {'f':>8}  {'v':>8} {'f':>8}  {'v':>8} {'f':>8}  {'v':>8} {'f':>8}  "
        f"{'(max err)':>10} {'(max err)':>10}"
    )
    print("-" * 137)
    for source, stats_list in sim_stats_by_source.items():
        n = len(stats_list)
        row = {k: np.mean([s[k] for s in stats_list]) for k in stats_list[0]}
        print(
            f"{source:<25} {n:>4}  "
            f"{row['dc_verts']:>8.0f} {row['dc_faces']:>8.0f}  "
            f"{row['dv_verts']:>8.0f} {row['dv_faces']:>8.0f}  "
            f"{row['sc_verts']:>8.0f} {row['sc_faces']:>8.0f}  "
            f"{row['sv_verts']:>8.0f} {row['sv_faces']:>8.0f}  "
            f"{row['bbox_diff_coll']:>10.6f} {row['bbox_diff_vis']:>10.6f}"
        )

    print(f"\nMeshes saved to: {OUTPUT_DIR}")

    # --- Part 3: Viser viewer ---
    server = viser.ViserServer()
    current_handles: list[viser.SceneNodeHandle] = []
    source_names = sorted(set(d["source"] for d in viewer_data))
    filtered_indices = list(range(len(viewer_data)))

    def load_object(slider_pos: int):
        nonlocal current_handles
        for handle in current_handles:
            handle.remove()
        current_handles.clear()

        d = viewer_data[filtered_indices[slider_pos]]
        for method, (x, color, label) in VISER_CONFIG.items():
            mesh = d[method]
            if mesh is None:
                continue
            normalized = normalize_mesh(mesh)
            offset = np.array([[x, 0.0, 0.0]])
            handle = server.scene.add_mesh_simple(
                name=f"/meshes/{method}",
                vertices=(normalized.vertices + offset).astype(np.float32),
                faces=normalized.faces,
                color=color,
                flat_shading=True,
            )
            current_handles.append(handle)
            label_handle = server.scene.add_label(
                name=f"/labels/{method}",
                text=f"{label} (v={len(mesh.vertices)}, f={len(mesh.faces)}, g={d['groups'][method]})",
                position=(x, 0.0, 0.6),
            )
            current_handles.append(label_handle)

        info_text.value = f"{d['source']} / {d['obj_name']}"

    with server.gui.add_folder("Controls"):
        source_dropdown = server.gui.add_dropdown("Source", ["All"] + source_names)
        object_slider = server.gui.add_slider("Object", min=0, max=len(viewer_data) - 1, step=1, initial_value=0)
        info_text = server.gui.add_text("Object ID", initial_value="", disabled=True)

    @source_dropdown.on_update
    def _(_event: viser.GuiEvent):
        nonlocal filtered_indices
        selected = source_dropdown.value
        filtered_indices = [i for i, d in enumerate(viewer_data) if selected == "All" or d["source"] == selected]
        object_slider.max = len(filtered_indices) - 1
        object_slider.value = 0
        load_object(0)

    @object_slider.on_update
    def _(_event: viser.GuiEvent):
        load_object(int(object_slider.value))

    load_object(0)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
