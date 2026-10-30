"""dex1b v0.4.1 scene descriptions."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

_DIR = fetch(
    ["data/scene_description/dex1b/v0.4.1/small_world_tabletop_mini.msgpack"], REVISION, HF_REPO_ID, GRASPKIT_HOME
)
SMALL_WORLD_TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.4.1/small_world_tabletop_mini.msgpack"
