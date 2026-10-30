"""Tests for `graspkit.isaac.worker`."""

import numpy as np
import pytest

from graspkit.isaac.worker import pack_array, unpack_array


class TestArrayTransport:
    @pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.bool_])
    def test_numpy_version_independent_transport(self, dtype: type):
        value = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=dtype)[:, ::2]
        packet = pack_array(value)
        assert isinstance(packet[2], bytes)
        actual = unpack_array(packet)
        assert actual.dtype == value.dtype and actual.flags.writeable
        np.testing.assert_array_equal(actual, value)
