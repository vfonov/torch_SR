from time import gmtime, strftime

import numpy as np

try:
    from minc2_simple import minc2_file, minc2_xfm, minc2_dim, minc2_error
    _HAVE_MINC2 = True
except ImportError:
    _HAVE_MINC2 = False

from .geo import decompose, compose

#: Recognised NIfTI filename suffixes (checked case-insensitively).
_NIFTI_SUFFIXES = (".nii", ".nii.gz")
#: Recognised MINC2 filename suffix.
_MINC_SUFFIX = ".mnc"


def _require_minc2():
    """Raise a clear error if ``minc2_simple`` isn't installed.

    Called at the top of every function below that touches ``minc2_file``,
    ``minc2_xfm``, ``minc2_dim`` or ``minc2_error`` -- importing this module
    must succeed without ``minc2_simple`` installed; only actually reading
    or writing a MINC file should fail.
    """
    if not _HAVE_MINC2:
        raise ImportError(
            "MINC2 I/O requires 'minc2_simple'; install it, e.g. "
            "pip install -e minc2-simple/python/")


def format_history(argv):
    """Create a MINC-style history entry from a command-line argv list."""
    stamp = strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp + (' '.join(argv))


def hdr_to_affine(hdr):
    """Convert MINC dimension headers to a 4×4 voxel-to-world affine matrix."""
    _require_minc2()
    rot    = np.zeros((3, 3))
    scales = np.zeros((3, 3))
    start  = np.zeros(3)

    ax = np.array([h.id for h in hdr])

    for i in range(3):
        aa = np.where(ax == (i + 1))[0][0]  # DIM_X=1, DIM_Y=2, DIM_Z=3
        if hdr[aa].have_dir_cos:
            rot[:, i] = hdr[aa].dir_cos
        else:
            rot[i, i] = 1

        scales[i, i] = hdr[aa].step
        start[i]     = hdr[aa].start

    origin = rot @ start
    out = np.eye(4)
    out[0:3, 0:3] = rot @ scales
    out[0:3, 3]   = origin
    return out


def affine_to_dims(aff, shape):
    """Convert a 4×4 affine matrix to a MINC dimension list."""
    _require_minc2()
    start, step, dir_cos = decompose(aff)
    if len(shape) == 3:
        dims = [
            minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i],
                      have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3, i]))
            for i in range(3)
        ]
    elif len(shape) == 4:
        dims = [
            minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i],
                      have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3, i]))
            for i in range(3)
        ] + [
            minc2_dim(id=minc2_file.MINC2_DIM_VEC, length=shape[3],
                      start=0, step=1, have_dir_cos=False, dir_cos=[0, 0, 0])
        ]
    else:
        raise ValueError(f"Unsupported number of dimensions: {len(shape)}")
    return dims


def load_minc_volume(fname, as_byte=False):
    """Load a MINC volume and return (data_tensor, voxel_to_world_matrix)."""
    _require_minc2()
    mm = minc2_file(fname)
    mm.setup_standard_order()
    d   = mm.load_complete_volume_tensor(
              minc2_file.MINC2_UBYTE if as_byte else minc2_file.MINC2_DOUBLE)
    aff = np.asmatrix(hdr_to_affine(mm.representation_dims()))
    mm.close()
    return d, aff


def load_minc_volume_np(fname, as_byte=False, dtype=None):
    """Load a MINC volume into a numpy array and return (data, voxel_to_world_matrix)."""
    _require_minc2()
    mm = minc2_file(fname)
    mm.setup_standard_order()
    if as_byte:
        dtype = 'uint8'
    elif dtype is None:
        dtype = 'float64'
    d   = mm.load_complete_volume(dtype)
    aff = np.asmatrix(hdr_to_affine(mm.representation_dims()))
    mm.close()
    return d, aff


def save_minc_volume(fn, data, aff, ref_fname=None, history=None):
    """Save a numpy array or tensor as a MINC file."""
    _require_minc2()
    dims = affine_to_dims(aff, data.shape)
    out  = minc2_file()
    if data.dtype == np.uint8:
        out.define(dims, minc2_file.MINC2_UBYTE,  minc2_file.MINC2_UBYTE)
    elif data.dtype == np.uint16:
        out.define(dims, minc2_file.MINC2_USHORT, minc2_file.MINC2_USHORT)
    elif data.dtype == np.int8:
        out.define(dims, minc2_file.MINC2_BYTE,   minc2_file.MINC2_BYTE)
    elif data.dtype == np.int16:
        out.define(dims, minc2_file.MINC2_SHORT,  minc2_file.MINC2_SHORT)
    elif data.dtype == np.float32:
        out.define(dims, minc2_file.MINC2_SHORT,  minc2_file.MINC2_FLOAT)
    elif data.dtype == np.float64:
        out.define(dims, minc2_file.MINC2_SHORT,  minc2_file.MINC2_DOUBLE)
    else:
        raise TypeError(f"Unsupported dtype: {data.dtype}")

    out.create(fn)

    if ref_fname is not None:
        ref = minc2_file(ref_fname)
        out.copy_metadata(ref)

    if history is not None:
        try:
            old_history = out.read_attribute("", "history") + "\n"
        except minc2_error:
            old_history = ""
        out.write_attribute("", "history", old_history + history)

    out.setup_standard_order()
    if isinstance(data, np.ndarray):
        out.save_complete_volume(np.ascontiguousarray(data))
    else:
        out.save_complete_volume_tensor(data)
    out.close()


def resample_volume(in_data, in_v2w, out_shape, out_v2w, order=1, fill=0.0):
    """Resample a volume to a different sampling grid using scipy."""
    import scipy.ndimage

    xyz_to_zyx = np.array([[0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])

    full_xfm = xyz_to_zyx @ np.linalg.inv(in_v2w) @ out_v2w @ xyz_to_zyx
    new_data = scipy.ndimage.affine_transform(
        in_data, full_xfm, output_shape=out_shape,
        order=order, mode='constant', cval=fill)
    return new_data, out_v2w


def uniformize_volume(data, v2w, tolerance=0.1, order=1, step=1.0):
    """Resample a volume to uniform isotropic voxel spacing if needed."""
    start, step_, dir_cos = decompose(v2w)

    if np.any(np.abs(step_ - step) > tolerance):
        xyz_to_zyx = np.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]])
        new_shape = np.ceil(np.array(data.shape) * step_[[2, 1, 0]]).astype(int)
        new_start = start - step_ * 0.5 + np.ones(3) * step * 0.5
        new_v2w   = compose(new_start, np.ones(3) * step, dir_cos)
        return resample_volume(data, v2w, new_shape, new_v2w, order=order, fill=0.0)
    else:
        return data, v2w


def load_nl_xfm(fn):
    """Load a non-linear-only transform from a .xfm file."""
    _require_minc2()
    x = minc2_xfm(fn)
    if x.get_n_concat() == 1 and x.get_n_type(0) == minc2_xfm.MINC2_XFM_LINEAR:
        raise ValueError("Transform is linear, not non-linear")

    _identity = np.asmatrix(np.identity(4))
    _eps = 1e-6
    if (x.get_n_type(0) == minc2_xfm.MINC2_XFM_LINEAR
            and x.get_n_type(1) == minc2_xfm.MINC2_XFM_GRID_TRANSFORM):
        assert np.linalg.norm(_identity - np.asmatrix(x.get_linear_transform(0))) <= _eps
        grid_file, grid_invert = x.get_grid_transform(1)
    elif x.get_n_type(0) == minc2_xfm.MINC2_XFM_GRID_TRANSFORM:
        grid_file, grid_invert = x.get_grid_transform(0)
    else:
        raise ValueError("Unsupported transform type")

    grid, v2w = load_minc_volume(grid_file, as_byte=False)
    return grid, v2w, grid_invert


def load_lin_xfm(fn):
    """Load a linear-only transform from a .xfm file."""
    _require_minc2()
    x = minc2_xfm(fn)
    if x.get_n_concat() == 1 and x.get_n_type(0) == minc2_xfm.MINC2_XFM_LINEAR:
        return np.asmatrix(x.get_linear_transform())
    raise ValueError("Transform is not purely linear")


# ---------------------------------------------------------------------------
# Format-agnostic entry points -- dispatch on filename extension between the
# MINC2 backend above and the NIfTI backend in ``sr.minc.nifti``. Prefer
# these from CLI code; use the ``*_minc_volume*`` functions directly only
# when MINC is genuinely required (e.g. reading a .xfm's grid file).
# ---------------------------------------------------------------------------

def _is_nifti(fname) -> bool:
    name = str(fname).lower()
    return any(name.endswith(suffix) for suffix in _NIFTI_SUFFIXES)


def _is_minc(fname) -> bool:
    return str(fname).lower().endswith(_MINC_SUFFIX)


def _unsupported_extension(fname):
    return ValueError(
        f"Unrecognised volume file extension for {str(fname)!r}; expected "
        f"one of {_MINC_SUFFIX}, {', '.join(_NIFTI_SUFFIXES)}")


def load_volume(fname, as_byte=False):
    """Load a volume and return (data_tensor, voxel_to_world_matrix).

    Dispatches on ``fname``'s extension: ``.mnc`` to :func:`load_minc_volume`,
    ``.nii``/``.nii.gz`` to ``sr.minc.nifti.load_nifti_volume``.
    """
    if _is_nifti(fname):
        from .nifti import load_nifti_volume
        return load_nifti_volume(fname, as_byte=as_byte)
    if _is_minc(fname):
        return load_minc_volume(fname, as_byte=as_byte)
    raise _unsupported_extension(fname)


def load_volume_np(fname, as_byte=False, dtype=None):
    """Load a volume into a numpy array and return (data, voxel_to_world_matrix).

    Dispatches on ``fname``'s extension: ``.mnc`` to
    :func:`load_minc_volume_np`, ``.nii``/``.nii.gz`` to
    ``sr.minc.nifti.load_nifti_volume_np``.
    """
    if _is_nifti(fname):
        from .nifti import load_nifti_volume_np
        return load_nifti_volume_np(fname, as_byte=as_byte, dtype=dtype)
    if _is_minc(fname):
        return load_minc_volume_np(fname, as_byte=as_byte, dtype=dtype)
    raise _unsupported_extension(fname)


def save_volume(fn, data, aff, ref_fname=None, history=None):
    """Save a numpy array or tensor as a volume file.

    Dispatches on ``fn``'s extension: ``.mnc`` to :func:`save_minc_volume`,
    ``.nii``/``.nii.gz`` to ``sr.minc.nifti.save_nifti_volume``. ``ref_fname``
    is only forwarded when it is the same format as ``fn`` -- MINC's
    ``copy_metadata`` needs an actual MINC reference, and a NIfTI reference
    would otherwise crash a MINC save (or silently mean nothing to a NIfTI
    one) on a cross-format conversion.
    """
    if _is_nifti(fn):
        from .nifti import save_nifti_volume
        if ref_fname is not None and not _is_nifti(ref_fname):
            ref_fname = None
        return save_nifti_volume(fn, data, aff, ref_fname=ref_fname,
                                 history=history)
    if _is_minc(fn):
        if ref_fname is not None and not _is_minc(ref_fname):
            ref_fname = None
        return save_minc_volume(fn, data, aff, ref_fname=ref_fname,
                                history=history)
    raise _unsupported_extension(fn)
