"""CoACD collision + manifold/ACVD visual for dex_diffuser meshes, in coacd-acvd Docker via process_dexdiffuser_mp.sh.

Env vars:
    INPUT_DIR:    mounted dex_diffuser data root (e.g. /data/input)
    OUTPUT_DIR:   mounted output directory (e.g. /data/output)
    ACVD_BIN:     path to ACVD binary (default: /workspace/ACVD/bin/ACVD)
    SOURCES:      comma-separated source list (default: egad,kit,contactdb,ycb)
    NUM_WORKERS:  number of parallel workers (default: 1)
"""

import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path

import coacd
import numpy as np
import trimesh

INPUT_DIR = Path(os.environ["INPUT_DIR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
ACVD_BIN = Path(os.environ.get("ACVD_BIN", "/workspace/ACVD/bin/ACVD"))
SOURCES = os.environ.get("SOURCES", "egad,kit,contactdb,ycb").split(",")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "1"))

COACD_THRESHOLD = 0.05
MANIFOLD_LEVEL_SET = 0.1
ACVD_VERTEX_COUNT = 2000
ACVD_GRADATION = 1.5
MIN_VOLUME = 1e-10


def discover_objects() -> list[tuple[str, str, Path]]:
    """Return list of (source, name, stl_path) for all objects."""
    entries = []
    for source in SOURCES:
        source_dir = INPUT_DIR / source
        if not source_dir.is_dir():
            continue
        for obj_dir in sorted(source_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            stl_files = list(obj_dir.glob("*.stl"))
            if not stl_files:
                continue
            entries.append((source, obj_dir.name, stl_files[0]))
    return entries


def process_object(entry: tuple[str, str, Path]) -> str:
    source, name, stl_path = entry
    out_dir = OUTPUT_DIR / source / name
    collision_path = out_dir / "collision.glb"
    visual_path = out_dir / "visual.glb"

    if collision_path.exists() and visual_path.exists():
        return f"skip {source}/{name}"

    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(stl_path), force="mesh")
    assert len(mesh.faces) > 0 and len(mesh.vertices) > 0

    # Normalize to unit scale before CoACD so MIN_VOLUME=1e-10 is a meaningful filter
    verts = np.array(mesh.vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)) / 2)
    mesh.vertices = (verts - center) / length

    # CoACD convex decomposition
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=COACD_THRESHOLD, decimate=True)
    mesh_parts = [trimesh.Trimesh(vs, fs) for vs, fs in parts]
    mesh_parts = [m for m in mesh_parts if abs(m.volume) > MIN_VOLUME]

    # Scale parts back to original world space
    for m in mesh_parts:
        m.vertices = m.vertices * length + center

    # Collision: scene of convex parts
    if not collision_path.exists():
        scene = trimesh.Scene({f"part_{i}": m for i, m in enumerate(mesh_parts)})
        with open(collision_path, "wb") as f:
            f.write(scene.export(file_type="glb"))

    # Visual: concatenate → manifold → ACVD simplify
    if not visual_path.exists():
        merged = trimesh.util.concatenate(mesh_parts)
        manifold_verts, manifold_faces = coacd.run_manifold(
            coacd.Mesh(merged.vertices, merged.faces),
            level_set=MANIFOLD_LEVEL_SET,
        )
        manifold_mesh = trimesh.Trimesh(manifold_verts, manifold_faces)

        with tempfile.TemporaryDirectory() as tmp:
            manifold_obj = os.path.join(tmp, "manifold.obj")
            manifold_mesh.export(manifold_obj)
            subprocess.run(
                [
                    str(ACVD_BIN),
                    manifold_obj,
                    str(ACVD_VERTEX_COUNT),
                    str(ACVD_GRADATION),
                    "-o",
                    f"{tmp}/",
                    "-of",
                    "simplified.ply",
                    "-m",
                    "1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            smooth_path = os.path.join(tmp, "smooth_simplified.ply")
            if os.path.exists(smooth_path):
                os.unlink(smooth_path)
            simplified = trimesh.load(os.path.join(tmp, "simplified.ply"), force="mesh", process=False)

        simplified.visual = trimesh.visual.ColorVisuals()
        with open(visual_path, "wb") as f:
            f.write(trimesh.Trimesh(simplified.vertices, simplified.faces).export(file_type="glb"))

    return f"done {source}/{name}"


def worker_fn(entry: tuple[str, str, Path]) -> str:
    source, name, _ = entry
    # Suppress CoACD C++ stdout/stderr per-worker
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    result = process_object(entry)
    os.dup2(old_stderr, 2)
    os.close(devnull)
    os.close(old_stderr)
    return result


def main():
    entries = discover_objects()
    print(f"Found {len(entries)} objects, processing with {NUM_WORKERS} workers")

    if NUM_WORKERS <= 1:
        for idx, entry in enumerate(entries):
            result = process_object(entry)
            if (idx + 1) % 10 == 0 or idx == len(entries) - 1:
                print(f"  [{idx + 1}/{len(entries)}] {result}")
    else:
        done = 0
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            for result in pool.imap_unordered(worker_fn, entries):
                done += 1
                if done % 10 == 0 or done == len(entries):
                    print(f"  [{done}/{len(entries)}] {result}")

    collision_count = sum(1 for _ in OUTPUT_DIR.rglob("collision.glb"))
    visual_count = sum(1 for _ in OUTPUT_DIR.rglob("visual.glb"))
    print(f"Done. Collision: {collision_count}, Visual: {visual_count}")


if __name__ == "__main__":
    main()
