import logging
import os
from pathlib import Path

from upath import UPath

logger = logging.getLogger("graspkit")

PROJECT_ROOT = (Path(__file__).parent.parent.parent).resolve()
ASSETS_ROOT = PROJECT_ROOT / "assets"

# Single source of data/logs across worktrees (works even pip-installed); use link() to symlink into the project.
GRASPKIT_HOME = Path.home() / ".graspkit"
DATA_ROOT = GRASPKIT_HOME / "data"
LOG_ROOT = GRASPKIT_HOME / "logs"
MUJOCO_EVAL_ROOT = DATA_ROOT / "mujoco_eval"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def link(target: Path = PROJECT_ROOT) -> None:
    """Symlink ~/.graspkit/{data,logs} into ``target`` (default: repo root) for local browsing."""
    for name in ("data", "logs"):
        dst = target / name
        if not dst.exists():
            dst.symlink_to(GRASPKIT_HOME / name)
            logger.info(f"Created symlink: {dst} -> {GRASPKIT_HOME / name}")


# Robot assets come from robokit; scenes/objects come from graspkit.assets. `import graspkit` fetches nothing.

S3_ROOT = UPath(os.environ.get("GRASPKIT_S3_ROOT", "s3://graspkit"))

HF_REPO_ID = "openrobokit/graspkit"
HF_ROOT = UPath("hf://datasets/openrobokit/graspkit@main/")

ROBOT_COLOR = (0.52, 0.76, 0.90)
OBJECT_COLOR = (0.87, 0.53, 0.64)
TABLE_COLOR = (1.0, 1.0, 1.0)
