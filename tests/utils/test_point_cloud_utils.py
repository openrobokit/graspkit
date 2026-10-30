"""Tests for `graspkit.utils.point_cloud_utils`."""

import numpy as np
import pytest

from graspkit.utils.point_cloud_utils import depth_to_point_cloud, masked_depth_to_point_cloud, process_point_cloud


def _table_plus_cluster() -> np.ndarray:
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(-0.1, 0.1, 30), np.linspace(-0.1, 0.1, 30))
    table = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1)  # dense plane at z=0
    cluster = rng.uniform([0.03, 0.03, 0.04], [0.06, 0.06, 0.07], size=(600, 3))  # raised blob
    return np.concatenate([table, cluster], axis=0).astype(np.float32)


class TestDepthToPointCloud:
    def test_depth_to_point_cloud(self):
        # fx=fy=100, principal point at origin, depth_scale=0.001 (mm->m).
        intrinsics = np.array([100.0, 100.0, 0.0, 0.0, 2, 2, 0.001])
        depth = np.array([[1000, 2000], [0, 0]], dtype=np.uint16)  # two valid pixels (row 0)
        pts = depth_to_point_cloud(depth, intrinsics)

        assert pts.shape == (2, 3)  # only z>0 pixels kept
        assert pts.dtype == np.float32
        # pixel (row0,col0): z=1.0, x=(0-0)*z/fx=0, y=0
        assert np.allclose(pts[0], [0.0, 0.0, 1.0])
        # pixel (row0,col1): z=2.0, x=(1-0)*z/fx=0.02, y=0
        assert np.allclose(pts[1], [0.02, 0.0, 2.0])

    def test_masked_depth_to_point_cloud_keeps_selected_pixel(self):
        intrinsics = np.array([100.0, 100.0, 0.0, 0.0, 2, 2, 0.001])
        depth = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        mask = np.array([[False, True], [False, False]])

        pts = masked_depth_to_point_cloud(depth, intrinsics, mask)

        np.testing.assert_allclose(pts, [[0.02, 0.0, 2.0]])


class TestProcessPointCloud:
    def test_process_point_cloud_removes_table_keeps_cluster(self):
        pytest.importorskip("open3d")  # process_point_cloud needs open3d (xarm/rby1 extra)
        points = _table_plus_cluster()
        eye = np.eye(4, dtype=np.float32)
        pts = process_point_cloud(points, eye, workspace_center=(0.0, 0.0, 0.0), sample_size=4096)

        assert pts.dtype == np.float32
        assert len(pts) > 0
        assert len(pts) <= 4096
        assert (pts[:, 2] > 0.02).all()  # table plane (z~0) removed, only the raised cluster remains
        assert (np.abs(pts[:, 0]) < 0.15).all() and (np.abs(pts[:, 1]) < 0.15).all()  # inside crop

    def test_process_point_cloud_param_divergence(self):
        pytest.importorskip("open3d")  # process_point_cloud needs open3d (xarm/rby1 extra)
        # xhand_right preset (z_filter strips the bottom layer) keeps <= ability_right preset on the same cloud.
        points = _table_plus_cluster()
        eye = np.eye(4, dtype=np.float32)
        n_ability = len(process_point_cloud(points, eye, (0.0, 0.0, 0.0), segment_distance_threshold=0.008))
        n_xhand = len(
            process_point_cloud(
                points, eye, (0.0, 0.0, 0.0), segment_distance_threshold=0.010, z_filter_min_offset=0.01
            )
        )
        assert n_xhand <= n_ability

    def test_process_point_cloud_empty_outside_crop(self):
        pytest.importorskip("open3d")  # process_point_cloud needs open3d (xarm/rby1 extra)
        points = _table_plus_cluster()
        eye = np.eye(4, dtype=np.float32)
        pts = process_point_cloud(points, eye, workspace_center=(10.0, 10.0, 0.0))  # nothing in crop
        assert pts.shape == (0, 3)
        assert pts.dtype == np.float32
