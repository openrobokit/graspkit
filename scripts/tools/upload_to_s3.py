"""
Copy files to S3 or a local directory, preserving their project-relative paths.

Usage:
uv run scripts/tools/upload_to_s3.py logs/train/xarm_ability_ycb_1_pose_1_scale_cvae_pointnet_local_mlp_0.02_kl_0.5_obj_0.1_grasp/checkpoints/iter_100000.pt
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
import os
from dataclasses import dataclass

import tyro
from fsspec.callbacks import TqdmCallback

from graspkit.constants import PROJECT_ROOT, S3_ROOT
from graspkit.utils.io_utils import build_filesystem

logger = logging.getLogger("graspkit")


@dataclass
class Args:
    local_path: tyro.conf.Positional[str]
    """Path to the file or directory to upload"""

    remote_root: str = str(S3_ROOT)
    """S3 URI or local destination root"""


def upload_to_s3(args: Args):
    local_path = os.path.normpath(os.path.abspath(args.local_path))

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File or directory {local_path} does not exist")

    rel_path = os.path.relpath(local_path, PROJECT_ROOT)
    remote_path = os.path.join(args.remote_root, rel_path)

    remote_fs = build_filesystem(args.remote_root)
    remote_fs.makedirs(os.path.dirname(remote_path), exist_ok=True)
    remote_fs.upload(
        local_path,
        remote_path,
        recursive=True,
        callback=TqdmCallback(tqdm_kwargs={"desc": "Uploading"}),
    )
    logger.info(f"Uploaded to {remote_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    upload_to_s3(args)
