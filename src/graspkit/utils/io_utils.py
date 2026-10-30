import logging
import os
import shutil
from functools import lru_cache

import dotenv

dotenv.load_dotenv()

import fsspec
import trimesh
from fsspec.callbacks import TqdmCallback

from graspkit.constants import GRASPKIT_HOME, HF_ROOT, S3_ROOT

logger = logging.getLogger("graspkit")


def _build_s3_filesystem() -> fsspec.AbstractFileSystem:
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


def _build_hf_filesystem() -> fsspec.AbstractFileSystem:
    token = os.getenv("HF_TOKEN")
    return fsspec.filesystem("hf", token=token)


def ensure_local_filepath(filepath: str, local_project_root: str = str(GRASPKIT_HOME), max_retries: int = 3) -> str:
    if filepath.startswith("s3://"):
        fs = _build_s3_filesystem()
        local_filepath = os.path.join(local_project_root, os.path.relpath(filepath, str(S3_ROOT)))
    elif filepath.startswith("hf://"):
        fs = _build_hf_filesystem()
        local_filepath = os.path.join(local_project_root, os.path.relpath(filepath, str(HF_ROOT)))
    else:
        return filepath

    if not os.path.exists(local_filepath):
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

        for attempt in range(max_retries):
            try:
                fs.download(
                    filepath,
                    local_filepath,
                    recursive=True,
                    callback=TqdmCallback(tqdm_kwargs={"desc": f"Downloading (attempt {attempt + 1}/{max_retries})"}),
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
                if os.path.exists(local_filepath):
                    try:
                        if os.path.isdir(local_filepath):
                            shutil.rmtree(local_filepath)
                        else:
                            os.remove(local_filepath)
                    except OSError:
                        pass

        if not os.path.exists(local_filepath):
            raise FileNotFoundError(f"Failed to download file to {local_filepath}")

    return local_filepath


def build_filesystem(path: str) -> fsspec.AbstractFileSystem:
    if path.startswith("s3://"):
        return _build_s3_filesystem()
    return fsspec.filesystem("file", auto_mkdir=True)


@lru_cache(maxsize=1024)
def get_cached_trimesh_from_file(mesh_path: str) -> trimesh.Trimesh:
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(str(GRASPKIT_HOME), mesh_path)
    return trimesh.load(mesh_path, force="mesh", process=False)
