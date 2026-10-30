"""SAPIEN's native robot materials and textured table, exported for Isaac so its frames match ManiSkill's."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

_REL_DIR = "data/isaac/appearance"
PATH: Path = fetch([f"{_REL_DIR}/*"], REVISION, HF_REPO_ID, GRASPKIT_HOME) / _REL_DIR
