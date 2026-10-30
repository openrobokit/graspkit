"""Rebuild objaverse_dex1b collision.glb via re-download + CoACD; no graspkit import, so CPU pods skip a full `uv sync`.

(local validation, processes 3 objects into a throwaway dir)
uv run --no-project \
  --with coacd --with trimesh --with fast-simplification --with numpy \
  --with fsspec --with s3fs --with tyro --with objaverse \
  python scripts/stages/asset_prep/process_objaverse.py \
  --output_dir /tmp/processed_v2 --limit 3

(cluster) run one copy per pod; pods are interchangeable — just add more to go faster.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per CoACD worker; must precede coacd import

import contextlib
import multiprocessing
import os.path
import random
import time
from dataclasses import dataclass

import coacd
import fsspec
import numpy as np
import objaverse
import trimesh
import tyro


@dataclass
class Config:
    input_dir: str
    """Existing processed glbs; their filenames enumerate the uids to reprocess."""
    output_dir: str
    """Destination root; `<output_dir>/<uid>/collision.glb`. S3 or local path."""
    limit: int | None = None
    """Process at most this many uids after shuffling; for local/canary tests."""
    num_workers: int | None = None
    """Pool size; defaults to int(worker_multiplier * cpu_count)."""
    worker_multiplier: float = 2.0
    """Oversubscribe cores so CoACD fills the gaps while other workers download."""
    coacd_threshold: float = 0.05
    """CoACD concavity threshold (lower = finer decomposition)."""
    max_input_faces: int = 20000
    """Pre-simplify raw meshes above this face count to bound CoACD cost."""
    max_parts: int = 64
    """Keep at most this many (largest-volume) convex parts per object."""
    max_download_retries: int = 5
    """Per-uid objaverse download attempts before skip-and-log."""


def build_filesystem(path: str) -> fsspec.AbstractFileSystem:
    if not path.startswith("s3://"):
        return fsspec.filesystem("file")
    endpoint = (
        os.getenv("FSSPEC_S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("S3_ENDPOINT_URL")
        or os.getenv("AWS_S3_ENDPOINT")
    )
    storage_options = {
        "config_kwargs": {
            "read_timeout": 900,
            "connect_timeout": 180,
            "retries": {"max_attempts": 10, "mode": "adaptive"},
        }
    }
    if endpoint:
        storage_options["client_kwargs"] = {"endpoint_url": endpoint}
    return fsspec.filesystem("s3", **storage_options)


def detect_cpus() -> int:
    """Pod's CPU allotment from cgroup CFS quota (os.cpu_count ignores K8s limits); falls back to host count locally."""
    v2 = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(v2):
        quota, period = open(v2).read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    v1q, v1p = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(v1q) and os.path.exists(v1p):
        quota = int(open(v1q).read())
        if quota > 0:
            return max(1, round(quota / int(open(v1p).read())))
    return os.cpu_count() or 1


def load_uids(input_dir: str) -> list[str]:
    fs = build_filesystem(input_dir)
    paths = fs.ls(input_dir, detail=False)
    return sorted(os.path.basename(p).removesuffix(".glb") for p in paths if p.endswith(".glb"))


# Per-worker globals (set in the pool initializer); avoids re-deriving config in each task.
_CFG: Config
_FS: fsspec.AbstractFileSystem


def _init_worker(cfg: Config) -> None:
    global _CFG, _FS
    _CFG = cfg
    _FS = build_filesystem(cfg.output_dir)
    coacd.set_log_level("error")
    # Silence CoACD's C++ stderr per worker.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


def process_one(uid: str) -> str:
    out_path = os.path.join(_CFG.output_dir, uid, "collision.glb")
    if _FS.exists(out_path):
        return "skip"

    # Download with retry (objaverse hosting is flaky); skip-and-log on persistent failure.
    local_path = None
    for attempt in range(_CFG.max_download_retries):
        try:
            with contextlib.redirect_stdout(None):
                objs = objaverse.load_objects(uids=[uid], download_processes=1)
            local_path = objs.get(uid)
            if local_path:
                break
        except Exception as e:  # noqa: BLE001 - fault isolation: one bad uid must not kill the shard
            if attempt == _CFG.max_download_retries - 1:
                return f"download_fail {uid}: {e}"
            time.sleep(2**attempt)
    if not local_path:
        return f"download_missing {uid}"

    try:
        mesh = trimesh.load(local_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            return f"bad_mesh {uid}"

        # Normalize to unit size (scene description applies its own scale later).
        verts = np.asarray(mesh.vertices)
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)) / 2)
        mesh.vertices = (verts - center) / length

        if mesh.faces.shape[0] > _CFG.max_input_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=_CFG.max_input_faces)

        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=_CFG.coacd_threshold, decimate=True)
        meshes = [trimesh.Trimesh(vs, fs) for vs, fs in parts]
        meshes = [m for m in meshes if abs(m.volume) > 1e-10]
        if not meshes:
            return f"empty_decomp {uid}"
        if len(meshes) > _CFG.max_parts:
            meshes = sorted(meshes, key=lambda m: abs(m.volume), reverse=True)[: _CFG.max_parts]

        scene = trimesh.Scene({f"part_{i}": m for i, m in enumerate(meshes)})
        _FS.makedirs(os.path.dirname(out_path), exist_ok=True)
        with _FS.open(out_path, "wb") as f:
            f.write(scene.export(file_type="glb"))
        return "done"
    except Exception as e:  # noqa: BLE001 - fault isolation
        return f"process_fail {uid}: {e}"
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


def main(cfg: Config) -> None:
    uids = load_uids(cfg.input_dir)
    print(f"Found {len(uids)} uids in {cfg.input_dir}", flush=True)

    # Work-stealing: subtract everything already on S3, then shuffle so independent pods converge on the remainder.
    out_fs = build_filesystem(cfg.output_dir)
    done = (
        {os.path.basename(p.rstrip("/")) for p in out_fs.ls(cfg.output_dir, detail=False)}
        if out_fs.exists(cfg.output_dir)
        else set()
    )
    uids = [u for u in uids if u not in done]
    print(f"{len(done)} already done; {len(uids)} remaining", flush=True)
    random.Random(os.urandom(8)).shuffle(uids)
    if cfg.limit is not None:
        uids = uids[: cfg.limit]

    workers = cfg.num_workers or max(1, int(cfg.worker_multiplier * detect_cpus()))
    print(f"Processing {len(uids)} uids with {workers} workers", flush=True)

    counts: dict[str, int] = {}
    begin = time.time()
    with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, uids), start=1):
            key = result.split(" ", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if i % 50 == 0 or i == len(uids):
                rate = i / (time.time() - begin) * 60
                print(f"[{i}/{len(uids)}] {rate:.0f} obj/min {dict(sorted(counts.items()))}", flush=True)
            if not result.startswith(("done", "skip")):
                print(f"  {result}", flush=True)

    print(f"Done: {dict(sorted(counts.items()))}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
