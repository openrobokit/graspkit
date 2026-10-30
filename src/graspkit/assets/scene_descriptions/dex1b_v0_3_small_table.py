"""dex1b v0.3 small-table scene descriptions."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

_DIR = fetch(
    ["data/scene_description/dex1b/v0.3_small_table/tabletop_mini.msgpack"], REVISION, HF_REPO_ID, GRASPKIT_HOME
)
TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.3_small_table/tabletop_mini.msgpack"
