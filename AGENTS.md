# NLM MRI Super-Resolution (Manjon 2010) — Agent Guide

Python/PyTorch re-implementation of the Manjon & Coupe (2010) non-local-means MRI
super-resolution algorithm. `README.md` documents the layout, CLI usage and
provenance; `research.md` has the algorithm/C-kernel/MINC2 details. This file
covers only what an agent needs to work in the tree.

## Layout

Everything buildable lives under `upsample/`. Two interchangeable backends
implement the same `nlm_upsample()`:

- `upsample/sr/` — C-backed, the **numerical reference**. `nlm_core.c`/`.h` is a
  MEX-free CFFI port of the original `cMRegularizarNLM3D_V2.c`; `_nlm_build.py`
  builds it; `regularize.py` wraps it; `interpolate.py` (spline init, shared with
  the PyTorch path), `core.py`, `metrics.py`, `minc/` (MINC2 I/O + affine geometry).
- `upsample/sr_pytorch/` — pure PyTorch, GPU-capable. `regularize.py` vectorizes
  the offset search with `avg_pool3d` instead of looping per voxel;
  `noise_estimate.py`, `dwt.py`; `optimize.py` is an **experimental** Adam method,
  slower and not better in PSNR — not selected by default.
- `upsample/sr_numpy/` — **experimental** torch-free NumPy port of the
  `sr_pytorch` *denoising* path only (`regularize`, `noise_estimate`, `dwt`, the
  two `core` noise-level helpers); no SR pipeline. Backs `denoise.py --backend
  numpy` (CPU / `float64` only). Oracle: `sr_pytorch` on CPU — kept in step by
  `tests/test_numpy_backend.py`.
- `upsample/_legacy/` — vendored EZminc sources + `build_legacy.py`. An oracle for
  `sr_pytorch.noise_estimate` and `sr_pytorch.dwt` **only** — deliberately *not* an
  oracle for the NLM filter (EZminc shares the first voxel's sigma per pair; this
  project uses each centre's own). See `upsample/_legacy/ezminc/README.md`.
- `upsample/{tests,scripts,data}/`, and the CLIs `upsample.py`, `denoise.py`,
  `noise_estimate.py`.
- `legacy/demo/NLMUpsample/` — original Matlab/MEX reference, read-only:
  `NLMUpsample2.m`, `BSplineInterpolation.m`,
  `linux-mac code/cMRegularizarNLM3D_V2.c`. `legacy/demo/denoising/` is unrelated
  demo code, not used by this pipeline.
- `minc2-simple/` — submodule providing MINC2 I/O (`USAGE.md`, `example/python/`).

## Build & Test

All commands run from `upsample/`.

```bash
cd upsample
pip install -r requirements.txt   # torch, numpy, scipy, cffi
git submodule update --init       # minc2-simple, if not populated

python sr/_nlm_build.py           # C extension -> sr/_nlm_core*.so
python _legacy/build_legacy.py    # optional EZminc oracle; needs GSL, FFTW, MINC toolkit

pytest tests/                     # C-ext / oracle tests skip when not built
pytest tests/test_interpolate.py -v    # single file, needs no C extension
pytest --cov=sr tests/
```

The `pytorch` method and all three CLIs work without the C extension; `--method c`
requires it.

There is **no CI workflow and no black/flake8/mypy config** in this repo. Match
the surrounding code's style and run `pytest tests/` before committing.

## MINC2 (and optional NIfTI) I/O

All file I/O goes through `sr.minc.io`:

```python
from sr.minc.io import (
    load_minc_volume,       # -> (torch.Tensor [Z,Y,X], np.matrix 4x4 voxel->world)
    load_minc_volume_np,    # numpy variant; as_byte / dtype args
    save_minc_volume,       # (fn, data, aff, ref_fname=None, history=None)
    load_volume, load_volume_np, save_volume,  # same, dispatched by extension
    format_history,
    resample_volume, uniformize_volume,   # scipy-backed regridding — reuse these
)
```

`io.py` calls `setup_standard_order()` on every open/create, so axis order is
always **Z → Y → X** (numpy C-order, slowest to fastest). Volume tensors are
`(D, H, W)`, or `(N, C, D, H, W)` when batching for PyTorch ops. Catch
`minc2_error` around MINC calls.

**`minc2_simple` and `nibabel` are both optional.** `sr.minc.io` and
`sr.minc.nifti` guard their imports (`_HAVE_MINC2`/`_HAVE_NIBABEL`) so
importing either module — and therefore any CLI — succeeds with neither
package installed; only a function that actually needs one raises a clear
`ImportError` naming it. Prefer `load_volume`/`load_volume_np`/`save_volume`
in CLI code: they pick the backend from the filename's extension (`.mnc` vs
`.nii`/`.nii.gz`) so a script doesn't have to. Adding the equivalent guard
(`_require_xxx()`, called first thing in every function touching the
optional import) is required for any new function added to either module.

## Conventions

- Absolute imports within `sr` / `sr_pytorch`; no wildcards;
  `from __future__ import annotations` at the top of modules.
- Google-style docstrings; module docstrings cite `Manjon 2010`; annotate public
  signatures and note tensor shapes inline (`# (D, H, W)`).
- Raise specific built-ins with a clear message; custom exceptions in
  `sr/exceptions.py`. No bare `assert` in library code (fine in tests).
- `logger = logging.getLogger(__name__)`; iteration counts, convergence metrics
  and shapes at `INFO`, per-patch detail at `DEBUG`. Library modules must not
  configure handlers — that is the CLIs' job.

## Testing

- pytest only, `tests/test_<module>.py`, `@pytest.mark.parametrize` for variants.
- Fixtures in `tests/conftest.py`: `small_volume`, `chunk` / `chunk_mask` (real
  volumes from `tests/data/`), `legacy_output` (recorded EZminc oracle).
- Regenerate fixtures with `tests/regenerate_reference.py` (EZminc oracle) and
  `scripts/generate_test_blocks.py` (`data/psnr_factor_*.npz`).
  `scripts/benchmark_sr.py` compares `c` vs `pytorch` PSNR and wall time;
  `scripts/benchmark_denoise.py` times one `regularize` pass across the `c` /
  `pytorch` (CPU+CUDA) / `numpy` backends on a shared volume, `c` as the
  reference column, interior-only Δ (its `run_all` is reused by
  `tests/test_denoise_benchmark.py`). It defaults to `--factors 2 2 2` because
  the C wrapper has no `mean_preserve=False` and is the identity at `1 1 1`.

## Gotchas

- **Do not assume CUDA is unavailable.** Select the device once at the entry
  point via `torch.cuda.is_available()` and let it decide.
- `sr_pytorch` defaults to `float32` on CUDA and `float64` on CPU (matching the C
  reference). Noise / local-std helpers use reflect-padded `avg_pool3d` to match
  `scipy.ndimage.uniform_filter(mode='reflect')` — keep that padding when editing.
- Prefer batched tensor ops over per-voxel loops; avoid `.cpu()` / `.numpy()` in
  hot loops; `torch.no_grad()` for inference.
- **Never install missing packages** to work around an import error — stop and
  report what is missing. Deps are `upsample/requirements.txt`.
- Do not commit binaries or large volumes; the small `upsample/tests/data/*.mnc`
  are the only whitelisted exception.
- No repository-wide license: `sr/nlm_core.c` descends from GPL code and
  `_legacy/ezminc/` is separately licensed. See `COPYING.NLM` before moving or
  relicensing any file.
