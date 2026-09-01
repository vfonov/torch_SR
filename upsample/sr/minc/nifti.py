"""NIfTI (.nii / .nii.gz) I/O backend, via the optional ``nibabel`` package.

Reuses this project's own volume convention rather than nibabel's: arrays
are ``(D, H, W)`` = ``(Z, Y, X)`` (numpy C-order, slowest to fastest — see
``sr.minc.io``'s module docstring), while the returned ``voxel_to_world``
affine expects a voxel index in **(x, y, z)** order. That is exactly the
relationship ``sr.minc.io.resample_volume``/``uniformize_volume`` already
assume (they permute zyx-stored data through an ``xyz_to_zyx`` matrix before
applying an affine built for xyz-ordered indices) — MINC volumes have this
same "storage order is the transpose of what the affine expects" property.

nibabel's native array order is ``(X, Y, Z, ...)`` and ``img.affine`` also
expects voxel indices in that (x, y, z) order, so transposing the loaded
array to zyx and keeping ``img.affine`` unchanged reproduces this project's
``v2w`` convention with no further reconciliation — the geometry helpers in
``sr.minc.io`` work unchanged on NIfTI-sourced data.

Caveat, not solved here: NIfTI's affine is conventionally RAS+, while a
MINC file's world space isn't guaranteed to be any particular convention.
This has no effect on a single file's own round-trip or on resampling a
volume onto its own grid (everything this module and ``sr.minc.io`` do
today) — it would only matter if code later resampled a NIfTI volume onto a
MINC file's grid (or vice versa) while assuming both worlds are the same
physical frame.
"""

from __future__ import annotations

import numpy as np

try:
    import nibabel as nib
    _HAVE_NIBABEL = True
except ImportError:
    _HAVE_NIBABEL = False


def _require_nibabel():
    """Raise a clear error if ``nibabel`` isn't installed.

    Called at the top of every function below -- importing this module must
    succeed without ``nibabel`` installed; only actually reading or writing
    a NIfTI file should fail.
    """
    if not _HAVE_NIBABEL:
        raise ImportError(
            "NIfTI I/O requires 'nibabel'; install it: pip install nibabel")


def _to_zyx(arr: np.ndarray) -> np.ndarray:
    """(X, Y, Z, ...) -> (Z, Y, X, ...)."""
    if arr.ndim == 3:
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))
    if arr.ndim == 4:
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0, 3)))
    raise ValueError(f"Unsupported number of dimensions: {arr.ndim}")


def _to_xyz(arr: np.ndarray) -> np.ndarray:
    """(Z, Y, X, ...) -> (X, Y, Z, ...); inverse of :func:`_to_zyx`."""
    if arr.ndim == 3:
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))
    if arr.ndim == 4:
        return np.ascontiguousarray(np.transpose(arr, (2, 1, 0, 3)))
    raise ValueError(f"Unsupported number of dimensions: {arr.ndim}")


def load_nifti_volume_np(fname, as_byte=False, dtype=None):
    """Load a NIfTI volume into a numpy array and return (data, voxel_to_world_matrix)."""
    _require_nibabel()
    img = nib.load(str(fname))
    if as_byte:
        dtype = 'uint8'
    elif dtype is None:
        dtype = 'float64'
    data = _to_zyx(np.asarray(img.get_fdata(dtype=np.float64 if as_byte else dtype)))
    if as_byte:
        data = data.astype('uint8')
    aff = np.asmatrix(np.asarray(img.affine, dtype=np.float64))
    return data, aff


def load_nifti_volume(fname, as_byte=False):
    """Load a NIfTI volume and return (data_tensor, voxel_to_world_matrix).

    nibabel has no native tensor loader; this wraps
    :func:`load_nifti_volume_np` with ``torch.from_numpy``.
    """
    import torch

    data, aff = load_nifti_volume_np(fname, as_byte=as_byte)
    return torch.from_numpy(data), aff


def save_nifti_volume(fn, data, aff, ref_fname=None, history=None):
    """Save a numpy array or tensor as a NIfTI file.

    ``ref_fname`` is accepted for signature parity with
    ``sr.minc.io.save_minc_volume`` but has no NIfTI equivalent (there is no
    separate metadata to copy from a reference file) and is ignored.
    ``history``, if given, is best-effort recorded in the 80-byte
    ``descrip`` header field -- NIfTI has nothing resembling MINC's
    free-text history log, so it is truncated to fit.
    """
    _require_nibabel()

    arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    arr = _to_xyz(arr)

    img = nib.Nifti1Image(arr, np.asarray(aff, dtype=np.float64))
    if history is not None:
        img.header['descrip'] = history[:80]

    nib.save(img, str(fn))
