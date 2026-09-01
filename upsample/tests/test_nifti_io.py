"""Round-trip tests for the nibabel-backed NIfTI I/O in ``sr.minc.nifti``.

Skipped entirely when ``nibabel`` isn't installed (it is optional --
``pip install nibabel``).
"""

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from sr.minc.io import load_minc_volume_np
from sr.minc.nifti import load_nifti_volume_np, save_nifti_volume

_DATA = "tests/data/chunk.mnc"


def test_round_trip_matches_source_data_and_grid(tmp_path):
    data, aff = load_minc_volume_np(_DATA)

    out = tmp_path / "chunk.nii.gz"
    save_nifti_volume(str(out), data, aff)
    data2, aff2 = load_nifti_volume_np(str(out))

    assert data2.shape == data.shape
    np.testing.assert_allclose(data2, data, rtol=1e-5, atol=1e-3)

    # Same voxel-to-world convention: the affine maps a voxel index given
    # in (x, y, z) order to world coordinates, regardless of which backend
    # produced it (see sr.minc.nifti's module docstring). Compare where a
    # handful of voxel corners land in world space rather than asserting
    # the two 4x4 matrices are byte-identical.
    corners = np.array([
        [0, 0, 0, 1],
        [data.shape[2] - 1, 0, 0, 1],
        [0, data.shape[1] - 1, 0, 1],
        [0, 0, data.shape[0] - 1, 1],
    ], dtype=np.float64)
    world1 = (np.asarray(aff) @ corners.T).T
    world2 = (np.asarray(aff2) @ corners.T).T
    np.testing.assert_allclose(world2, world1, rtol=1e-5, atol=1e-3)


def test_as_byte_round_trips_as_uint8(tmp_path):
    data, aff = load_minc_volume_np(_DATA, as_byte=True)
    assert data.dtype == np.uint8

    out = tmp_path / "chunk_byte.nii.gz"
    save_nifti_volume(str(out), data, aff)
    data2, _ = load_nifti_volume_np(str(out), as_byte=True)

    assert data2.dtype == np.uint8
    np.testing.assert_array_equal(data2, data)
