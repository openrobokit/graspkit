"""dexgraspnet_official_splits object data."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

DIR: Path = (
    fetch(["data/object_data/dexgraspnet_official_splits/**"], REVISION, HF_REPO_ID, GRASPKIT_HOME)
    / "data/object_data"
    / "dexgraspnet_official_splits"
)
