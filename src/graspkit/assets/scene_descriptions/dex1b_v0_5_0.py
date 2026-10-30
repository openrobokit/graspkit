"""dex1b v0.5.0 scene descriptions."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

_DIR = fetch(
    [
        "data/scene_description/dex1b/v0.5.0/tabletop_mini.msgpack",
        "data/scene_description/dex1b/v0.5.0/wbc_tabletop_mini.msgpack",
        "data/scene_description/dex1b/v0.5.0/wbc_tabletop_small.msgpack",
        "data/scene_description/dex1b/v0.5.0/dexmate_tabletop_mini.msgpack",
    ],
    REVISION,
    HF_REPO_ID,
    GRASPKIT_HOME,
)
TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.5.0/tabletop_mini.msgpack"
DEXMATE_TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.5.0/dexmate_tabletop_mini.msgpack"
WBC_TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.5.0/wbc_tabletop_mini.msgpack"
WBC_TABLETOP_SMALL_PATH: Path = _DIR / "data/scene_description/dex1b/v0.5.0/wbc_tabletop_small.msgpack"
