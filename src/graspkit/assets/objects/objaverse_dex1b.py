"""objaverse_dex1b object meshes (extracted at import into DATA_ROOT/object_data/objaverse_dex1b)."""

from pathlib import Path

from graspkit.assets.objects import extract_object_archive

DATA_DIR: Path = extract_object_archive("objaverse_dex1b_v2.zip", "objaverse_dex1b")
