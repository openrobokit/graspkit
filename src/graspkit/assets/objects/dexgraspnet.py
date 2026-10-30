"""dexgraspnet object meshes (fetched + extracted at import into DATA_ROOT/object_data/dexgraspnet)."""

from pathlib import Path

from graspkit.assets.objects import extract_object_archive

DATA_DIR: Path = extract_object_archive("dexgraspnet.tar.gz", "dexgraspnet")
