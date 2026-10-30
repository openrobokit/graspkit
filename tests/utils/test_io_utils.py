"""Tests for `graspkit.utils.io_utils`."""

import pytest

from graspkit.utils.io_utils import build_filesystem


class TestBuildFilesystem:
    @pytest.mark.parametrize("as_uri", [False, True])
    def test_local_artifact_commit_creates_directories(self, tmp_path, as_uri):
        root = (tmp_path / "artifacts").as_uri() if as_uri else str(tmp_path / "artifacts")
        source = tmp_path / "rows.jsonl"
        source.write_bytes(b'{"success": true}\n')
        fs = build_filesystem(root)
        fs.put_file(str(source), f"{root}/results/chunk0/rows.jsonl")
        fs.pipe_file(f"{root}/results/chunk0/complete.json", b'{"episodes": 1}')
        assert fs.cat(f"{root}/results/chunk0/rows.jsonl") == source.read_bytes()
        assert fs.cat(f"{root}/results/chunk0/complete.json") == b'{"episodes": 1}'
