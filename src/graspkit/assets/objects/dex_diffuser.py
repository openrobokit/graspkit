"""dex_diffuser object meshes (fetched + extracted at import into DATA_ROOT/object_data/dex_diffuser)."""

from pathlib import Path

from graspkit.assets.objects import extract_object_archive

DATA_DIR: Path = extract_object_archive("dex_diffuser.zip", "dex_diffuser")
