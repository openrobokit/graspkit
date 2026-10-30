import logging
import os
from shutil import copyfileobj

from rich.progress import track

logger = logging.getLogger("graspkit")


def _ensure_directory(path: str):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def _truncate(text: str, length: int = 15) -> str:
    return text if len(text) <= length else text[: length - 3] + "..."


def _extract_zipfile(filename: str, extract_dir: str, remove_top_dir: bool = False):
    import zipfile  # lazy import

    if not zipfile.is_zipfile(filename):
        raise ValueError(f"{filename} is not a valid zip file.")

    with zipfile.ZipFile(filename) as zip:
        names = [info.filename for info in zip.infolist()]

        # check if the zip file contains a common top-level directory
        contain_top_dir = os.path.commonprefix(names).count("/") > 0

        for name in track(names, description=_truncate(os.path.basename(filename))):
            # don't extract absolute paths or ones with .. in them
            if name.startswith("/") or ".." in name:
                continue

            if remove_top_dir and contain_top_dir:
                targetpath = os.path.join(extract_dir, *name.split("/")[1:])
            else:
                targetpath = os.path.join(extract_dir, *name.split("/"))

            if not targetpath:
                continue
            _ensure_directory(targetpath)
            if not name.endswith("/"):
                # file
                with zip.open(name, "r") as source, open(targetpath, "wb") as target:
                    copyfileobj(source, target)


def _extract_tarfile(filename: str, extract_dir: str, remove_top_dir: bool = False):
    import tarfile  # lazy import

    with tarfile.open(filename, "r:*") as tar:
        members = tar.getmembers()
        names = [member.name for member in members if member.type == tarfile.REGTYPE]
        # check if the tar file contains a common top-level directory
        contain_top_dir = os.path.commonprefix(names).count("/") > 0
        for member in track(members, description=_truncate(os.path.basename(filename))):
            name = member.name
            # don't extract absolute paths or ones with .. in them
            if name.startswith("/") or ".." in name:
                continue
            if remove_top_dir and contain_top_dir:
                targetpath = os.path.join(extract_dir, *name.split("/")[1:])
            else:
                targetpath = os.path.join(extract_dir, *name.split("/"))
            if not targetpath:
                continue
            _ensure_directory(targetpath)
            if member.isfile():
                # file
                with tar.extractfile(member) as source, open(targetpath, "wb") as target:  # type: ignore
                    copyfileobj(source, target)


def extract_archive(filename: str, extract_dir: str, remove_top_dir: bool = False):
    """Extract archive file to a directory.

    Args:
        filename (str): Local path to the archive file.
        extract_dir (str): Directory to extract the archive file.
        remove_top_dir (bool, optional): Remove the top-level common-prefix directory when True. Defaults to False.
    """
    logger.info(f"Extracting {filename} to {extract_dir}")
    if os.path.splitext(filename)[-1] == ".zip":
        _extract_zipfile(filename, extract_dir, remove_top_dir)
    elif filename.endswith((".tar", ".tar.gz", ".tar.bz2", ".tar.xz")):
        _extract_tarfile(filename, extract_dir, remove_top_dir)
    else:
        raise NotImplementedError(f"Unsupported archive format: {filename}")


__all__ = ["extract_archive"]
