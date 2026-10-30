"""Object-mesh datasets, one module per source; importing one (opt-in) fetches its HF archive and exposes DATA_DIR."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import DATA_ROOT, GRASPKIT_HOME, HF_REPO_ID
from graspkit.utils.archive_utils import extract_archive


def extract_object_archive(archive_name: str, dest_subdir: str) -> Path:
    """Fetch+extract ``<archive_name>`` into ``dest_subdir``; a marker makes this a no-op after the first import."""
    dest = DATA_ROOT / "object_data" / dest_subdir
    marker = dest / f".extracted.{archive_name}"
    if marker.is_file():
        return dest
    archive = (
        fetch([f"data/object_data/{archive_name}"], REVISION, HF_REPO_ID, GRASPKIT_HOME)
        / "data/object_data"
        / archive_name
    )
    dest.mkdir(parents=True, exist_ok=True)
    extract_archive(str(archive), extract_dir=str(dest), remove_top_dir=True)
    marker.touch()
    return dest
