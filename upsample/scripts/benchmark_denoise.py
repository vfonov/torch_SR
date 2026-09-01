#!/usr/bin/env python3
"""Benchmark one NLM regularization pass across every available backend.

Runs the single ``regularize()`` call ``denoise.py`` makes -- ``v=3``,
``f=1`` -- on **one** volume with **one** shared uniform sigma, timing each
backend:

* ``c``                -- ``sr.regularize`` (compiled CFFI kernel), CPU
* ``pytorch/cpu``      -- ``sr_pytorch.regularize``, float64
* ``pytorch/cuda/f32`` -- ``sr_pytorch.regularize``, float32 (if CUDA present)
* ``pytorch/cuda/f64`` -- ``sr_pytorch.regularize``, float64 (if CUDA present)
* ``numpy``            -- ``sr_numpy.regularize`` (experimental torch-free), float64

The ``c`` (CFFI) kernel is the reference column for the reported timing
ratio and max abs difference.

``--factors`` defaults to ``2 2 2`` (block-mean correction on): the C wrapper
has no ``mean_preserve=False`` and with ``factors 1 1 1`` its correction
restores every voxel -- i.e. the C backend *cannot* do a plain denoise pass,
so it is skipped there and ``numpy`` becomes the reference.  With a real
factor every backend does the same work and the float64 ones agree to
~1e-13 on the interior (``pytorch/cuda/f32`` to ~float32 precision).

The max abs difference is measured on the **interior** only -- the ``v + f``
voxels next to each face are dropped, since that is the reach of the boundary
condition and the backends handle it differently (C skips out-of-bounds
search neighbours, the others reflect-pad).

Usage::

    # from /app/upsample
    python scripts/benchmark_denoise.py                    # synthetic 64^3
    python scripts/benchmark_denoise.py --size 96
    python scripts/benchmark_denoise.py --input tests/data/chunk.mnc
    python scripts/benchmark_denoise.py --factors 1 1 1    # numpy reference
    python scripts/benchmark_denoise.py --repeat 3 --no-cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_UPSAMPLE = Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))


V, F = 3, 1

#: Voxels within this distance of any face are dropped before the max abs
#: difference is measured: the search radius plus the patch half-size is the
#: reach of the boundary condition, and the backends handle it differently
#: (C skips out-of-bounds neighbours, the others reflect-pad).
BORDER = V + F


def _interior(arr: np.ndarray) -> np.ndarray:
    b = BORDER
    if min(arr.shape) <= 2 * b:
        return arr
    return arr[b:-b, b:-b, b:-b]


# ---------------------------------------------------------------------------
# Backend runners -- each returns (output_array, seconds)
# ---------------------------------------------------------------------------

def _time(fn, repeat: int):
    fn()  # warm-up (kernel load / CUDA context / allocator)
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def run_c(vol, sigma, factors, mean_preserve, repeat):
    # ``mean_preserve`` is accepted for a uniform runner signature; the C
    # kernel always applies the block-mean correction and has no toggle.
    del mean_preserve
    from sr.regularize import regularize

    return _time(lambda: regularize(vol, V, F, sigma, factors), repeat)


def run_numpy(vol, sigma, factors, mean_preserve, repeat):
    from sr_numpy.regularize import regularize

    return _time(
        lambda: regularize(vol, V, F, sigma, factors,
                           mean_preserve=mean_preserve),
        repeat,
    )


def run_torch(vol, sigma, factors, mean_preserve, repeat, device, dtype):
    import torch
    from sr_pytorch.regularize import regularize

    vol_t = torch.as_tensor(vol, dtype=dtype, device=device)
    sig_t = torch.as_tensor(sigma, dtype=dtype, device=device)

    def call() -> np.ndarray:
        out = regularize(vol_t, V, F, sig_t, factors,
                         mean_preserve=mean_preserve)
        if device == "cuda":
            torch.cuda.synchronize()
        return out.detach().cpu().numpy()

    return _time(call, repeat)


# ---------------------------------------------------------------------------
# Plan / run
# ---------------------------------------------------------------------------

def available_backends(factors, no_cuda: bool, no_c: bool) -> list[dict]:
    """The backends to benchmark, in report order, each a spec dict."""
    import torch

    plain_denoise = tuple(factors) == (1, 1, 1)
    backends: list[dict] = []

    from sr.regularize import have_c_ext
    if no_c:
        pass
    elif not have_c_ext():
        backends.append({"label": "c", "kind": "skip",
                         "why": "C extension not built (python sr/_nlm_build.py)"})
    elif plain_denoise:
        backends.append({"label": "c", "kind": "skip",
                         "why": "sr.regularize has no mean_preserve=False; "
                                "factors=(1,1,1) is the identity"})
    else:
        backends.append({"label": "c", "kind": "c"})

    backends.append({"label": "pytorch/cpu", "kind": "torch",
                     "device": "cpu", "dtype": torch.float64})

    if torch.cuda.is_available() and not no_cuda:
        backends.append({"label": "pytorch/cuda/f32", "kind": "torch",
                         "device": "cuda", "dtype": torch.float32})
        backends.append({"label": "pytorch/cuda/f64", "kind": "torch",
                         "device": "cuda", "dtype": torch.float64})
    elif not no_cuda:
        backends.append({"label": "pytorch/cuda", "kind": "skip",
                         "why": "CUDA not available"})

    backends.append({"label": "numpy", "kind": "numpy"})
    return backends


def run_all(vol: np.ndarray, sigma: np.ndarray, factors=(2, 2, 2),
            repeat: int = 1, no_cuda: bool = False,
            no_c: bool = False) -> list[dict]:
    """Run every available backend on the same ``vol`` / ``sigma`` / ``factors``.

    Returns a list of result dicts with keys ``label``, ``time`` (seconds or
    ``None`` if skipped), ``out`` (array or ``None``) and ``why`` (skip
    reason).  Reused by ``tests/test_denoise_benchmark.py``.
    """
    factors = tuple(factors)
    mean_preserve = factors != (1, 1, 1)
    results: list[dict] = []
    for spec in available_backends(factors, no_cuda, no_c):
        if spec["kind"] == "skip":
            results.append({"label": spec["label"], "time": None,
                            "out": None, "why": spec["why"]})
            continue
        if spec["kind"] == "c":
            out, dt = run_c(vol, sigma, factors, mean_preserve, repeat)
        elif spec["kind"] == "numpy":
            out, dt = run_numpy(vol, sigma, factors, mean_preserve, repeat)
        else:
            out, dt = run_torch(vol, sigma, factors, mean_preserve, repeat,
                                spec["device"], spec["dtype"])
        results.append({"label": spec["label"], "time": dt, "out": out,
                        "why": None})
    return results


#: Backend used as the reference column, in order of preference.  The C
#: (CFFI) kernel is the numerical reference for the project; fall back only
#: when it did not run.
_REFERENCE_ORDER = ("c", "numpy", "pytorch/cpu")


def _reference_label(results: list[dict]) -> str | None:
    ran = {r["label"] for r in results if r["out"] is not None}
    return next((lab for lab in _REFERENCE_ORDER if lab in ran), None)


def format_table(results: list[dict], shape, factors, repeat: int) -> str:
    ref_label = _reference_label(results)
    ref = next((r["out"] for r in results
                if r["label"] == ref_label and r["out"] is not None), None)
    base = next((r["time"] for r in results
                 if r["label"] == ref_label and r["time"] is not None), None)

    tag = f"vs {ref_label}" if ref_label else "vs ref"
    lines = [
        f"volume {tuple(shape)}  v={V} f={F} factors={tuple(factors)}  "
        f"repeat={repeat} (best of)   reference: {ref_label}",
        "",
        f"{'backend':<17}  {'time (s)':>10}  {tag:>9}  "
        f"{'max|Δ| ' + tag + ' (interior)':>26}",
        "-" * 69,
    ]
    for r in results:
        if r["time"] is None:
            lines.append(f"{r['label']:<17}  {'skipped':>10}  "
                         f"{'':>9}  {r['why']}")
            continue
        speed = f"{base / r['time']:>8.2f}x" if base else ""
        if ref is not None and r["out"] is not None:
            d = float(np.abs(_interior(r["out"].astype(np.float64))
                             - _interior(ref)).max())
            dcol = f"{d:>26.3e}"
        else:
            dcol = f"{'':>26}"
        lines.append(f"{r['label']:<17}  {r['time']:>10.4f}  {speed:>9}  {dcol}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_volume(args) -> np.ndarray:
    if args.input:
        from sr.minc.io import load_volume_np
        vol, _ = load_volume_np(str(args.input))
        return np.ascontiguousarray(vol, dtype=np.float64)
    rng = np.random.default_rng(0)
    n = args.size
    return rng.random((n, n, n)) * 256.0


def build_sigma(vol: np.ndarray, voxel: bool = False) -> np.ndarray:
    """The shared sigma every backend is benchmarked with.

    Uniform by default (``denoise.py``'s ``--auto-sigma-global``, and the
    case in which C and the others agree on the interior); per-voxel with
    ``voxel=True``.  Reused by ``tests/test_denoise_benchmark.py``.
    """
    from sr_numpy.core import _noise_level

    level = _noise_level(vol) + 1.0
    if not voxel:
        level = np.full_like(vol, float(level.mean()))
    return np.ascontiguousarray(level, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=None,
                   help="Volume file to use (.mnc/.nii); default is synthetic")
    p.add_argument("--size", type=int, default=64,
                   help="Cube side for the synthetic volume")
    p.add_argument("--factors", nargs=3, type=int, default=[2, 2, 2],
                   metavar=("FZ", "FY", "FX"),
                   help="Block-mean-correction factors; 1 1 1 skips the C "
                        "backend (it cannot do a plain denoise pass)")
    p.add_argument("--repeat", type=int, default=1,
                   help="Timed runs per backend (best is reported)")
    p.add_argument("--no-cuda", action="store_true", help="Skip pytorch/cuda")
    p.add_argument("--no-c", action="store_true", help="Skip the C backend")
    p.add_argument("--sigma-voxel", action="store_true",
                   help="Use the per-voxel noise map instead of one uniform "
                        "level (C then differs from the others on the "
                        "interior too -- see README)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vol = make_volume(args)
    sigma = build_sigma(vol, args.sigma_voxel)
    factors = tuple(args.factors)

    results = run_all(vol, sigma, factors, args.repeat,
                      args.no_cuda, args.no_c)
    print()
    print(format_table(results, vol.shape, factors, args.repeat))
    print()


if __name__ == "__main__":
    main()
