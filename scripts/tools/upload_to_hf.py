"""
Upload files to huggingface repository.

Usage:
python scripts/tools/upload_to_hf.py data/scene_description/dex1b/v1.json
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
import os
from dataclasses import dataclass
from typing import Optional

import tyro
from huggingface_hub import HfApi

from graspkit.constants import HF_REPO_ID, PROJECT_ROOT

logger = logging.getLogger("graspkit")


@dataclass
class Args:
    local_path: tyro.conf.Positional[str]
    """Path to the file or directory to upload"""

    repo_id: str = str(HF_REPO_ID)
    """Remote root path"""

    path_in_repo: Optional[str] = None
    """Remote path to upload to. Defaults to local_path's path relative to the project root."""


def upload_to_remote(args: Args) -> None:
    """Upload local file or directory to huggingface repo, mirroring the local path by default."""
    local_path = os.path.normpath(os.path.abspath(args.local_path))

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File or directory {local_path} does not exist")

    rel_path_unix = args.path_in_repo or os.path.relpath(local_path, PROJECT_ROOT).replace(os.sep, "/")

    api = HfApi()
    if os.path.isdir(local_path):
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=rel_path_unix,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"upload folder {rel_path_unix}",
        )
    else:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path_unix,
            repo_id=args.repo_id,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"upload file {rel_path_unix}",
        )
    logger.info(f"Uploaded to {args.repo_id}/{rel_path_unix}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    upload_to_remote(args)
