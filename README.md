# torch_SR

A PyTorch reimplementation of the non-local-means (NLM) MRI super-resolution
algorithm of Manjon et al. (2010), with a C-backed reference path ported
from the authors' original Matlab/MEX code and a pure-PyTorch, GPU-capable
path. The package also exposes the individual NLM regularization and
noise-estimation stages as standalone command-line tools, for denoising or
parameter inspection independent of the upsampling pipeline.

## References

- Manjón, J. V., Coupé, P., Buades, A., Fonov, V., Collins, D. L., & Robles, M.
  (2010). "MRI superresolution using self-similarity and image priors."
  *International Journal of Biomedical Imaging*, 2010. Also published as
  "Non-local MRI upsampling," *Medical Image Analysis* 14(6), 784–792. The
  algorithm this project ports; the PDF is checked out locally as
  `manjon_2010_upsample.pdf`.
- Coupé, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., & Barillot, C.
  (2008). "An Optimized Blockwise Non Local Means Denoising Filter for 3D
  Magnetic Resonance Images." *IEEE Transactions on Medical Imaging* 27(4),
  425–441. The blockwise NLM denoising filter (MBONLM) shipped as
  reference/demo code in `legacy/demo/denoising/`, not used by this
  project's own pipeline.
- Coupé, P., Prima, S., Hellier, P., Kervrann, C., & Barillot, C. (2008). "3D
  Wavelet Sub-Bands Mixing for Image Denoising." *International Journal of
  Biomedical Imaging*, 2008. The wavelet-mixing demo code under
  `legacy/demo/denoising/wavelet/`, likewise reference-only.
- Coupé et al. (2009), the Rician noise-level estimation method reproduced
  by `upsample/sr_pytorch/noise_estimate.py` and, as an oracle, by the
  vendored EZminc `noise_estimate` program under `upsample/_legacy/ezminc/`.

Copyright on the original NLM method and its C/Matlab distribution is held
by Jose V. Manjón and Pierrick Coupé (2010). `COPYING.NLM` is the
authoritative provenance and license note for the code in this repository
that descends from their work; the repository carries no single
repository-wide license, only per-file and per-algorithm notices.

## Repository layout

| Path | Role |
|---|---|
| `legacy/demo/NLMUpsample/` | Original Matlab/MEX reference: `NLMUpsample2.m` (top-level driver), `BSplineInterpolation.m`, `cMRegularizarNLM3D_V2.c` (the C MEX kernel). Read-only. |
| `legacy/demo/denoising/` | Unrelated reference/demo code from the same author distribution: `ANLM3D.m`, `MBONLM3D.c` (blockwise NLM denoising), and `wavelet/` (3-D wavelet sub-band mixing). Not used by this project's pipeline. |
| `legacy/demo/test3D.m`, `test3Dnoisy.m`, `testReal.m` | Matlab demo/test drivers for the original code, with accompanying sample data (`T2LR.hdr`/`.img`, `t1_icbm_normal_1mm_pn0_rf0.rawb`). |
| `upsample/sr/` | C-backed implementation. `nlm_core.c`/`.h` is a MEX-free CFFI port of `cMRegularizarNLM3D_V2.c`; `core.py` is the full `nlm_upsample()` pipeline; `interpolate.py` is the spline initial interpolation shared with the PyTorch path; `regularize.py` is the CFFI wrapper; `metrics.py` computes PSNR; `minc/` holds MINC2 I/O and affine geometry helpers, plus optional NIfTI I/O (`nifti.py`, needs `nibabel`) dispatched to by extension from `io.py`. |
| `upsample/sr_pytorch/` | Pure-PyTorch, GPU-capable reimplementation. `core.py` is a drop-in `nlm_upsample()` replacement; `regularize.py` is the vectorized NLM kernel; `optimize.py` is an experimental Adam-based alternative; `noise_estimate.py` is a Rician/Gaussian noise-level estimator; `dwt.py` is a one-level 3-D Daubechies-4 discrete wavelet transform used by the noise estimator. |
| `upsample/sr_numpy/` | **Experimental** torch-free NumPy port of `sr_pytorch`'s *denoising* path only (`regularize.py`, `noise_estimate.py`, `dwt.py`, the two `core.py` noise-level helpers) — no super-resolution pipeline. Backs `denoise.py --backend numpy`; CPU / `float64` only, numerically matching the `sr_pytorch` CPU path (`tests/test_numpy_backend.py`). |
| `upsample/_legacy/ezminc/` | Vendored EZminc CFFI sources (`mincnlm`, `minc_anlm`, `noise_estimate`), built via `build_legacy.py` into `_ezminclegacy`/`_ezminc_anlm`. Used only to validate `sr_pytorch.noise_estimate` and `sr_pytorch.dwt` against a byte-level oracle — explicitly **not** an oracle for this project's own NLM filter, which follows a different convention (each voxel's own local sigma vs. EZminc's shared first-voxel sigma per pair). See `upsample/_legacy/ezminc/README.md`. |
| `upsample/_legacy/compat/` | Stand-in headers (`volume_io.h`, `time_stamp.h`, `ParseArgv.h`) letting the vendored EZminc sources build without the full MINC toolkit. |
| `upsample/tests/` | pytest suite: `test_interpolate.py` (no C extension needed), `test_regularize.py`/`test_core.py` (need the C extension), `test_pytorch.py`/`test_pytorch_regularize.py`/`test_denoise.py` (the PyTorch path), `test_numpy_backend.py` (the `sr_numpy` port vs. the `sr_pytorch` CPU path), `test_denoise_benchmark.py` (cross-backend denoise timing/agreement), `test_dwt.py`/`test_noise_estimate.py`/`test_ezminc_backend.py` (checked against the EZminc oracle), `test_psnr.py`. `reference.py`/`regenerate_reference.py` record EZminc program outputs into `tests/reference/legacy_ezminc.{npz,json}`. |
| `upsample/data/` | PSNR regression fixtures: `psnr_factor_{2,3,5,7}.npz`. |
| `upsample/scripts/` | `benchmark_sr.py` (C vs. PyTorch backend timing/PSNR), `benchmark_denoise.py` (one `regularize` pass timed across the `c` / `pytorch` (CPU+CUDA) / `numpy` backends on one shared volume, `c` as reference, interior-only max abs difference; defaults to `--factors 2 2 2` since the C wrapper is the identity at `1 1 1`), `generate_test_blocks.py` (regenerates the `data/*.npz` fixtures), `nlm_upsample.py` (an earlier copy of the top-level upsampling CLI). |
| `minc2-simple/` | Git submodule providing MINC2 volume I/O and PyTorch spatial-transform examples (`USAGE.md`, `example/python/`). |
| `AGENTS.md` | Development conventions: directory layout, build/test commands, MINC2 I/O patterns, code style, logging, CI requirements. |
| `research.md` | Detailed technical notes: coordinate conventions, the C kernel's memory layout, algorithm steps, convergence logic, the CFFI build, the MINC2 API, and a fuller CLI/test reference. Its directory diagram predates the project being nested under `torch_SR/` and shows paths without that prefix. |
| `COPYING.NLM` | Provenance and license note for the NLM code. |

## Algorithm

The pipeline (`sr.core.nlm_upsample` / `sr_pytorch.core.nlm_upsample`)
follows five stages:

1. Normalize the input volume's intensities onto `[0, 256]`.
2. Build an initial high-resolution estimate by placing each low-resolution
   sample at its corresponding high-resolution block centre and filling the
   rest with cubic B-spline interpolation (`sr.interpolate.initial_interpolation`).
3. Estimate a per-voxel noise level from the local intensity variance.
4. Iteratively regularize the high-resolution estimate with a non-local-means
   update — a weighted average over a `(2v+1)³` search neighbourhood of
   `(2f+1)³` patches, weighted by patch similarity and the estimated noise —
   with block-mean preservation against the low-resolution input, until a
   convergence-ratio threshold (`--tol`) or a maximum iteration count is
   reached.
5. Rescale back to the input's original intensity range.

Two interchangeable backends implement stage 4:

- **`c`** (`sr.core`, `sr.regularize`): the CFFI-compiled, MEX-free port of
  the original per-voxel C kernel. Used as the numerical reference for the
  PyTorch path.
- **`pytorch`** (`sr_pytorch.core`, `sr_pytorch.regularize`): a
  GPU-capable reimplementation that vectorizes the search over offsets with
  `avg_pool3d` rather than looping per voxel. Defaults to `float32` on CUDA
  and `float64` on CPU (matching the C reference); the noise/local-std
  helpers use reflect-padded `avg_pool3d`, matching
  `scipy.ndimage.uniform_filter(mode='reflect')`.

A third, **experimental** method, `optimize` (`sr_pytorch.optimize`), skips
the closed-form NLM iteration and instead optimizes an additive
high-resolution correction field with Adam against a downsample-consistency
objective. It is documented as slower than the `pytorch` method and not
better in PSNR, and is not selected by default.

## Installation

From `upsample/`:

```bash
pip install -r requirements.txt        # torch, numpy, scipy, cffi

# Build the C extension (once; produces sr/_nlm_core.*.so)
python sr/_nlm_build.py

# Build the vendored EZminc oracle extensions (once; needs GSL, the
# toolkit's static FFTW, and an installed MINC toolkit). Only required
# for the oracle-comparison tests (test_ezminc_backend.py, test_dwt.py,
# test_noise_estimate.py).
python _legacy/build_legacy.py
```

`minc2-simple` (MINC2 volume I/O) is a git submodule; initialize it with
`git submodule update --init` if it is not already populated. The `pytorch`
method and the standalone denoising/regularization/noise-estimation CLIs
work without the C extension; `--method c` (or its default selection when
the extension is present) requires it.

Both volume I/O backends are optional and independent: `sr.minc.io` and its
CLIs import cleanly without either installed, and only fail — with a clear
message — when you actually try to load or save the corresponding format.
`minc2_simple` (`.mnc`) is installed as above; NIfTI (`.nii`/`.nii.gz`)
support needs `pip install nibabel`, not listed in `requirements.txt`
since it's optional in the same way.

## Usage

All commands are run from `upsample/` and operate on MINC2 (`.mnc`) or,
with `nibabel` installed, NIfTI (`.nii`/`.nii.gz`) volumes — chosen by file
extension, no flag needed.

**`upsample.py`** — full super-resolution pipeline:

```bash
python upsample.py input.mnc output.mnc --factors 2 1 1 --method pytorch
```

Key options: `--factors FZ FY FX` (integer upsampling factors, numpy `Z Y X`
axis order, default `2 1 1`), `--method c|pytorch|optimize` (default: `c` if
built, else `pytorch`), `--v`/`--f` (search radius / patch half-size),
`--tol` (convergence ratio, default `1.2`), `--max-iter`, `--initial-only`
(spline interpolation only, no NLM), `--device`, `--n-steps`/`--lr` (for
`--method optimize`), `--verbose`.

**`denoise.py`** — single NLM pass, exposing the `regularize` kernel
directly: plain denoising by default, or regularization of an already
spline-interpolated volume when given `--factors`:

```bash
python denoise.py hr.mnc hr_denoised.mnc --strength 2.0
python denoise.py hr.mnc hr_reg.mnc --factors 2 1 1
```

The noise level comes from exactly one of four mutually exclusive flags:
`--auto-sigma-global` (one uniform level from `sr_pytorch.noise_estimate`,
**the default** when none is given), `--auto-sigma-voxel` (a per-voxel
local-variance estimate), `--sigma VALUE` (that uniform level) or
`--sigma-file FILE` (a per-voxel noise map as a MINC2 volume). `--gauss`
switches the automatic uniform estimate from Rician (default) to Gaussian.

`--strength` scales the noise level however it was obtained (`0` is the
identity). `--floor-mode legacy|scale` selects the fixed noise floor
(`legacy`, default) or one derived from the volume's dynamic range (`scale`,
affine-equivariant). `--factors FZ FY FX` are the upsampling factors the
input was interpolated with; the block-mean correction is enabled
automatically whenever they are not `1 1 1`, and `--no-mean-preserve` forces
it off. Combining `--mean-preserve` with `--factors 1 1 1` is rejected: every
block would be a single voxel, so the correction would restore the input
exactly and discard the filtering pass.

`--backend pytorch` (default) uses `sr_pytorch` and honours `--device` /
`--dtype`. `--backend numpy` selects the **experimental** torch-free
`sr_numpy` path — CPU / `float64` only (`--device` / `--dtype` are rejected),
numerically matching the PyTorch CPU path.

**`noise_estimate.py`** — Rician/Gaussian noise-level or SNR estimate:

```bash
python noise_estimate.py input.mnc --snr --verbose
```

Prints the noise level by default, the SNR with `--snr`, both with
`--noise --snr`; `--mask` overrides the built-in k-means object detection;
`--bins` sets the histogram bin count (default `2000`); `--gauss` selects
the Gaussian noise model.

## Testing

```bash
cd upsample
pytest tests/
```

Tests requiring the C extension or the EZminc oracle extensions skip
automatically when those are not built. `pytest tests/test_core.py -v` runs
a single file; `pytest --cov=sr tests/` reports coverage. `data/psnr_factor_*.npz`
are the fixed regression fixtures used by `test_psnr.py`, generated by
`scripts/generate_test_blocks.py`; `scripts/benchmark_sr.py` reports PSNR
and wall-clock time for the `c` and `pytorch` backends against them.

## License and provenance

`COPYING.NLM` documents the code's provenance in detail. In summary:

- `upsample/sr/nlm_core.c` is a direct, MEX-free port of the original
  GPL-licensed C kernel, `cMRegularizarNLM3D_V2.c`.
- `upsample/sr_pytorch/regularize.py` is an independent PyTorch
  reimplementation of the same computation — whole-volume tensor operations
  rather than a per-voxel loop — and differs from the C kernel by
  convention as well as by expression: the C kernel uses the first voxel's
  sigma and local mean for both directions of a symmetric update, while
  this implementation uses each centre's own.
- Both are derivatives of the GPL-licensed original by descent.
- `upsample/_legacy/ezminc/` is separately licensed vendored code (Coupé et
  al.'s EZminc project); see `upsample/_legacy/ezminc/README.md` for its
  own provenance and license notes.

There is no repository-wide license file; anyone redistributing this
repository should resolve licensing against these per-file notices first.

## Agentic AI disclosure

Authorship in this repository splits by origin:

- The original C sources are human-authored: the EZminc `mincnlm`,
  `minc_anlm`, and `noise_estimate` code vendored under
  `upsample/_legacy/ezminc/` (Coupé et al.), and the Manjon/Coupé
  Matlab+MEX distribution under `legacy/demo/`, including the C kernel
  `cMRegularizarNLM3D_V2.c` that `upsample/sr/nlm_core.c` ports.
- All Python code and all tests in this repository — `upsample/sr/`,
  `upsample/sr_pytorch/`, the top-level CLI scripts (`upsample.py`,
  `denoise.py`, `noise_estimate.py`), `upsample/scripts/`,
  the Python build/wrapper code under `upsample/_legacy/`, and the
  `upsample/tests/` suite — were written by Claude Code.
- This `README.md` was likewise written by Claude Code, from a survey of
  the repository's existing files.
