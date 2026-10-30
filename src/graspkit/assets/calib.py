"""Camera calib files (repo-root ``calib/``) from graspkit's HF dataset; deploy config, outside ``data/``."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID


def calib_file(relpath: str) -> Path:
    return fetch([f"calib/{relpath}"], REVISION, HF_REPO_ID, GRASPKIT_HOME) / "calib" / relpath
