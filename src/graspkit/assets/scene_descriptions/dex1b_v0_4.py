"""dex1b v0.4 scene descriptions."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID

_DIR = fetch(
    [
        "data/scene_description/dex1b/v0.4/floating_mini.msgpack",
        "data/scene_description/dex1b/v0.4/tabletop_mini.msgpack",
        "data/scene_description/dex1b/v0.4/tabletop_dev.msgpack",
    ],
    REVISION,
    HF_REPO_ID,
    GRASPKIT_HOME,
)
FLOATING_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.4/floating_mini.msgpack"
TABLETOP_MINI_PATH: Path = _DIR / "data/scene_description/dex1b/v0.4/tabletop_mini.msgpack"
TABLETOP_DEV_PATH: Path = _DIR / "data/scene_description/dex1b/v0.4/tabletop_dev.msgpack"
