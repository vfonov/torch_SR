"""Equivalence tests for the experimental torch-free ``sr_numpy`` backend.

``sr_numpy`` is a NumPy port of ``sr_pytorch``'s denoising path.  Its
contract is: given the same inputs, it reproduces ``sr_pytorch`` on CPU /
``float64`` to tight floating-point tolerance (not bit-for-bit — the DWT's
FFT and the pooling reductions accumulate in a different order).  Every test
here pins one piece of that contract against the PyTorch reference.
"""

import importlib.util
import pathlib
import shutil
import sys

import numpy as np
import pytest
import torch

from sr_numpy import core as np_core
from sr_numpy import dwt as np_dwt
from sr_numpy import noise_estimate as np_ne
from sr_numpy import regularize as np_reg
from sr_pytorch import core as pt_core
from sr_pytorch import dwt as pt_dwt
from sr_pytorch import noise_estimate as pt_ne
from sr_pytorch import regularize as pt_reg

_UPSAMPLE = pathlib.Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))

_spec = importlib.util.spec_from_file_location(
    "upsample_denoise_cli", _UPSAMPLE / "denoise.py")
assert _spec is not None and _spec.loader is not None
_denoise_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_denoise_cli)

_DATA = pathlib.Path(__file__).resolve().parent / "data"


def _max_rel(result, expected):
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.abs(np.asarray(result) - expected).max()
                 / np.abs(expected).max())


@pytest.mark.parametrize("strength", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("floor_mode", ["legacy", "scale"])
@pytest.mark.parametrize("factors", [(1, 1, 1), (2, 2, 2)])
def test_regularize_matches_pytorch(small_volume, strength, floor_mode, factors):
    vol = small_volume.to(torch.float64).numpy() * 256.0
    sigma = np_core._noise_level(vol) + 1.0
    mean_preserve = factors != (1, 1, 1)

    out_np = np_reg.regularize(vol, 3, 1, sigma, factors,
                               mean_preserve=mean_preserve, strength=strength,
                               floor_mode=floor_mode)
    out_pt = pt_reg.regularize(torch.from_numpy(vol), 3, 1,
                               torch.from_numpy(sigma), factors,
                               mean_preserve=mean_preserve, strength=strength,
                               floor_mode=floor_mode).numpy()
    assert np.allclose(out_np, out_pt, rtol=1e-6, atol=1e-8)


def test_regularize_golden_sum():
    """The same seed/shape/params as
    ``test_pytorch_regularize.test_default_args_reproduce_pre_change_output``;
    the numpy backend must land on the PyTorch golden sum."""
    rng = np.random.default_rng(0)
    vol = rng.random((16, 16, 16)) * 256.0
    sigma = rng.random((16, 16, 16)) * 10.0
    out = np_reg.regularize(vol, 3, 1, sigma, factors=(2, 2, 2))
    assert out.sum() == pytest.approx(520529.6381982323, rel=1e-6)


def test_noise_level_matches_pytorch(small_volume):
    vol = small_volume.to(torch.float64).numpy()
    assert np.allclose(np_core._noise_level(vol),
                       pt_core._noise_level(torch.from_numpy(vol)).numpy(),
                       rtol=1e-9, atol=1e-12)
    assert np.allclose(np_core._local_std(vol),
                       pt_core._local_std(torch.from_numpy(vol)).numpy(),
                       rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("shape", [(16, 16, 16), (15, 17, 13), (20, 20, 20)])
def test_dwt_matches_pytorch(shape):
    rng = np.random.default_rng(1)
    vol = rng.random(shape)
    got = np_dwt.dwt_forward(vol)
    want = pt_dwt.dwt_forward(torch.from_numpy(vol))
    for g, w in zip(got, want):
        assert g.shape == tuple(w.shape)
        assert _max_rel(g, w.numpy()) < 1e-6


@pytest.mark.parametrize("gaussian", [True, False])
def test_noise_estimate_matches_pytorch(chunk, gaussian):
    vol = chunk.to(torch.float64).numpy()
    n_np = np_ne.noise_estimate(vol, gaussian=gaussian)
    n_pt = pt_ne.noise_estimate(chunk.to(torch.float64), gaussian=gaussian)
    assert n_np[0] == pytest.approx(float(n_pt[0]), rel=1e-5)
    assert n_np[1] == pytest.approx(float(n_pt[1]), rel=1e-5)


def test_noise_estimate_masked_matches_pytorch(chunk, chunk_mask):
    vol = chunk.to(torch.float64).numpy()
    mask = chunk_mask.numpy()
    n_np = np_ne.noise_estimate(vol, mask=mask)
    n_pt = pt_ne.noise_estimate(chunk.to(torch.float64),
                                mask=chunk_mask)
    assert n_np[0] == pytest.approx(float(n_pt[0]), rel=1e-5)
    assert n_np[1] == pytest.approx(float(n_pt[1]), rel=1e-5)


def test_cli_rejects_device_with_numpy_backend():
    with pytest.raises(SystemExit):
        _denoise_cli._parse_args(
            ["in.mnc", "out.mnc", "--backend", "numpy", "--device", "cuda"])
    with pytest.raises(SystemExit):
        _denoise_cli._parse_args(
            ["in.mnc", "out.mnc", "--backend", "numpy", "--dtype", "float32"])


def test_cli_numpy_backend_end_to_end(tmp_path):
    src = _DATA / "chunk.mnc"
    inp = tmp_path / "in.mnc"
    out = tmp_path / "out.mnc"
    shutil.copy(src, inp)
    sys.argv = ["denoise.py", str(inp), str(out), "--backend", "numpy",
                "--sigma", "5000"]
    _denoise_cli.main()
    assert out.exists()

    from sr.minc.io import load_volume_np
    data, _ = load_volume_np(str(out))
    assert np.isfinite(data).all()

    # Same explicit sigma through the PyTorch CPU path: outputs must agree.
    out_pt = tmp_path / "out_pt.mnc"
    sys.argv = ["denoise.py", str(inp), str(out_pt), "--backend", "pytorch",
                "--device", "cpu", "--sigma", "5000"]
    _denoise_cli.main()
    ref, _ = load_volume_np(str(out_pt))
    assert _max_rel(data, ref) < 1e-6
