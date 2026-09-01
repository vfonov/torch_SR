"""Tests for the optional-dependency guards and format dispatch in
``sr.minc.io`` and ``sr.minc.nifti``.

Neither ``minc2_simple`` nor ``nibabel`` should be required just to *import*
these modules -- only to actually load or save the corresponding format.
Both packages are installed in this dev environment (``minc2_simple``
separately from the unpopulated ``minc2-simple/`` submodule; ``nibabel`` as
an optional extra -- ``pip install nibabel``), so their absence is
simulated via ``sys.modules`` rather than relied on from the environment,
keeping these tests meaningful whether or not either package happens to be
installed.
"""

import importlib
import sys

import pytest

import sr.minc.io as io
import sr.minc.nifti as nifti

_DATA = "tests/data/chunk.mnc"


def _block_import(monkeypatch, module):
    """Make ``import <module>`` raise ImportError, without touching the
    real environment: a module set to ``None`` in ``sys.modules`` does
    this. Returns the real module object (or ``None``) so the caller can
    restore it -- monkeypatch only undoes ``setitem`` at test teardown,
    too late for a reload happening mid-test."""
    real = sys.modules.get(module)
    monkeypatch.setitem(sys.modules, module, None)
    return real


def _restore_import(module, real):
    if real is not None:
        sys.modules[module] = real
    else:
        sys.modules.pop(module, None)


# ---------------------------------------------------------------------------
# minc2_simple is optional
# ---------------------------------------------------------------------------

def test_io_module_imports_without_minc2_simple(monkeypatch):
    real = _block_import(monkeypatch, "minc2_simple")
    reloaded = importlib.reload(io)
    try:
        assert reloaded._HAVE_MINC2 is False
        with pytest.raises(ImportError, match="minc2_simple"):
            reloaded.load_minc_volume_np(_DATA)
    finally:
        _restore_import("minc2_simple", real)
        importlib.reload(io)  # restore real minc2_simple bindings
    assert io._HAVE_MINC2 is True


# ---------------------------------------------------------------------------
# nibabel is optional
# ---------------------------------------------------------------------------

def test_nifti_module_imports_without_nibabel(monkeypatch):
    real = _block_import(monkeypatch, "nibabel")
    reloaded = importlib.reload(nifti)
    try:
        assert reloaded._HAVE_NIBABEL is False
        with pytest.raises(ImportError, match="nibabel"):
            reloaded.load_nifti_volume_np("x.nii")
        with pytest.raises(ImportError, match="nibabel"):
            reloaded.save_nifti_volume("x.nii", None, None)
    finally:
        _restore_import("nibabel", real)
        importlib.reload(nifti)  # restore real nibabel bindings


# ---------------------------------------------------------------------------
# extension dispatch
# ---------------------------------------------------------------------------

def test_load_volume_np_dispatches_mnc_to_minc_backend():
    data_direct, aff_direct = io.load_minc_volume_np(_DATA)
    data_dispatch, aff_dispatch = io.load_volume_np(_DATA)
    assert (data_direct == data_dispatch).all()
    assert (aff_direct == aff_dispatch).all()


def test_load_volume_np_dispatches_nifti_to_nibabel_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sr.minc.nifti.load_nifti_volume_np",
        lambda fname, as_byte=False, dtype=None: calls.append(fname) or (1, 2))

    assert io.load_volume_np("x.nii") == (1, 2)
    assert io.load_volume_np("x.nii.gz") == (1, 2)
    assert calls == ["x.nii", "x.nii.gz"]


def test_save_volume_dispatches_nifti_to_nibabel_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sr.minc.nifti.save_nifti_volume",
        lambda fn, data, aff, ref_fname=None, history=None: calls.append(fn))

    io.save_volume("x.nii.gz", None, None)
    assert calls == ["x.nii.gz"]


def test_load_volume_np_rejects_unrecognized_extension():
    with pytest.raises(ValueError, match=r"\.mnc"):
        io.load_volume_np("x.foo")


def test_save_volume_rejects_unrecognized_extension():
    with pytest.raises(ValueError, match=r"\.mnc"):
        io.save_volume("x.foo", None, None)
