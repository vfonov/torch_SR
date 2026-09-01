#!/usr/bin/env python3
"""Filter a MINC2 (or NIfTI) volume with one non-local-means pass.

    python upsample/denoise.py input.mnc output.mnc [options]

Input/output format is chosen by extension: ``.mnc`` (needs
``minc2_simple``) or ``.nii``/``.nii.gz`` (needs the optional ``nibabel``
package -- ``pip install nibabel``). See ``sr.minc.io``.

A single call to ``regularize``, exposed directly.  By default this is a
plain per-voxel NLM filter (``--factors 1 1 1``, no block-mean correction);
pass ``--factors`` to enable the mean-preservation correction used when
post-processing an already spline-interpolated volume from the SR pipeline.

``--backend pytorch`` (default) uses ``sr_pytorch`` and honours ``--device``
/ ``--dtype``.  ``--backend numpy`` is an **experimental** torch-free path
(``sr_numpy``): CPU and ``float64`` only, numerically matching the PyTorch
CPU path.

Sigma sources
-------------
Exactly one of four mutually exclusive flags decides where the noise level
comes from:

* ``--auto-sigma-global`` — one uniform level estimated from the volume by
  ``sr_pytorch.noise_estimate.noise_estimate`` (Rician/Koay by default,
  ``--gauss`` for the Gaussian variant). **This is the default** when no
  sigma flag is given.
* ``--auto-sigma-voxel`` — a per-voxel estimate (``sr_pytorch.core``'s
  ``_noise_level``: local std, smoothed, halved).
* ``--sigma VALUE`` — that uniform level, on the volume's own intensity
  scale.
* ``--sigma-file FILE`` — a per-voxel noise map read from a volume file
  (``.mnc``, ``.nii``, ``.nii.gz``).

This sigma dispatch lives here, in the CLI, not inside ``regularize()`` —
see ``sr_pytorch/regularize.py``'s module docstring.

There is no per-voxel ``--mask`` support for ``--auto-sigma-global``, even
though ``noise_estimate`` itself accepts one; use ``upsample/noise_estimate.py
--mask`` to get a masked estimate and pass it back in with an explicit
``--sigma <value>``.

Mean preservation
-----------------
``--factors FZ FY FX`` are the upsampling factors the input was interpolated
with; the correction shifts each ``(fz, fy, fx)`` output block to match its
input block's mean. It is enabled automatically whenever ``--factors`` is
something other than ``1 1 1``, and can be forced off with
``--no-mean-preserve``. Requesting ``--mean-preserve`` *with* ``1 1 1`` is
rejected: every block would be a single voxel, so the correction would
restore the input exactly and the whole filtering pass would be discarded.

Examples
--------
    # default: uniform auto-estimated sigma (Rician), v=3 f=1, GPU
    python upsample/denoise.py hr.mnc hr_denoised.mnc

    # per-voxel auto-estimated sigma, smoothing doubled
    python upsample/denoise.py hr.mnc hr_denoised.mnc --auto-sigma-voxel \\
        --strength 2.0

    # explicit uniform sigma, CPU, affine-equivariant floor
    python upsample/denoise.py hr.mnc hr_denoised.mnc --sigma 5.0 --device cpu \\
        --floor-mode scale

    # post-process a volume interpolated with factors 2 1 1
    python upsample/denoise.py hr.mnc hr_reg.mnc --factors 2 1 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

_UPSAMPLE = Path(__file__).resolve().parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))


def _sigma(value: str) -> float:
    """``--sigma`` takes one uniform noise level, on the volume's own
    intensity scale. To have it estimated instead, use
    ``--auto-sigma-global`` or ``--auto-sigma-voxel``."""
    level = float(value)
    if not level > 0:
        raise argparse.ArgumentTypeError(
            "a noise level is positive; use --auto-sigma-global or "
            "--auto-sigma-voxel to estimate one")
    return level


def _resolve_mean_preserve(mean_preserve: Optional[bool],
                           factors: Tuple[int, int, int]) -> bool:
    """Mean preservation defaults to on only for non-trivial *factors*.

    With ``factors == (1, 1, 1)`` every block is a single voxel, so the
    correction reproduces the input exactly (see
    ``sr_pytorch.regularize``); defaulting it off keeps a plain
    ``denoise.py`` run an actual filtering pass.
    """
    if mean_preserve is None:
        return tuple(factors) != (1, 1, 1)
    return mean_preserve


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-pass non-local-means filtering of a MINC2 "
                    "volume (Manjon et al. 2010).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input volume (.mnc, .nii, .nii.gz)")
    p.add_argument("output", type=Path, help="Output volume (.mnc, .nii, .nii.gz)")

    p.add_argument("--v", type=int, default=3,
                   help="Search-neighbourhood radius; cost is cubic in this "
                        "and in nothing else")
    p.add_argument("--f", type=int, default=1,
                   help="Patch half-size (patch is (2f+1)^3 voxels)")

    strength_grp = p.add_mutually_exclusive_group()
    strength_grp.add_argument("--strength", type=float, default=1.0,
                              help="Multiplies the estimated noise level; 0 "
                                   "is exactly the identity")
    strength_grp.add_argument("--h", type=float, dest="strength",
                              default=argparse.SUPPRESS,
                              help=argparse.SUPPRESS)

    p.add_argument("--floor-mode", choices=["legacy", "scale"],
                   default="legacy",
                   help="'legacy' matches sr_pytorch.regularize's original "
                        "fixed noise floor; 'scale' derives it from the "
                        "volume's own dynamic range (affine-equivariant)")

    sigma_grp = p.add_mutually_exclusive_group()
    sigma_grp.add_argument("--auto-sigma-global", action="store_true",
                           help="estimate one uniform noise level from the "
                                "volume (the default when no sigma flag is "
                                "given)")
    sigma_grp.add_argument("--auto-sigma-voxel", action="store_true",
                           help="estimate a per-voxel noise level from the "
                                "volume's local variance")
    sigma_grp.add_argument("--sigma", type=_sigma, default=None,
                           metavar="VALUE",
                           help="use this uniform noise level, on the "
                                "volume's own intensity scale")
    sigma_grp.add_argument("--sigma-file", type=Path, default=None,
                           metavar="FILE",
                           help="Volume file (.mnc, .nii, .nii.gz) with a "
                                "per-voxel noise-level map (must match the "
                                "input's shape)")

    p.add_argument("--gauss", action="store_true",
                   help="--auto-sigma-global's noise model: Rician "
                        "(Koay-corrected) by default, Gaussian with this "
                        "flag")

    p.add_argument("--factors", nargs=3, type=int, default=[1, 1, 1],
                   metavar=("FZ", "FY", "FX"),
                   help="Upsampling factors the input was interpolated with; "
                        "enables the block-mean correction when not 1 1 1")
    mp_grp = p.add_mutually_exclusive_group()
    mp_grp.add_argument("--mean-preserve", dest="mean_preserve",
                        action="store_true", default=None,
                        help="Force the block-mean correction on (invalid "
                             "with --factors 1 1 1, where it is the identity)")
    mp_grp.add_argument("--no-mean-preserve", dest="mean_preserve",
                        action="store_false",
                        help="Force the block-mean correction off")

    p.add_argument("--backend", choices=["pytorch", "numpy"], default="pytorch",
                   help="'pytorch' (default) or the experimental torch-free "
                        "'numpy' path (CPU / float64 only)")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device, e.g. 'cuda'. Defaults to 'cuda' "
                        "if available, else 'cpu'")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="Floating-point dtype. Defaults to float32 on "
                        "CUDA, float64 on CPU")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress and parameter summary")
    return p


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.gauss and not args.auto_sigma_global:
        p.error("--gauss selects the noise model of the automatic uniform "
                "estimate; it is only meaningful with --auto-sigma-global")

    if args.backend == "numpy" and (args.device is not None
                                    or args.dtype is not None):
        p.error("--device / --dtype are PyTorch-only; the numpy backend "
                "always runs on CPU in float64")

    factors = tuple(args.factors)
    if args.mean_preserve and factors == (1, 1, 1):
        p.error("--mean-preserve with --factors 1 1 1 is the identity: every "
                "block is a single voxel, so the correction restores the "
                "input and discards the whole filtering pass. Pass real "
                "--factors, or drop --mean-preserve")

    return args


def _load_sigma_file(args: argparse.Namespace, shape):
    """Read and shape-check the ``--sigma-file`` map (a numpy array)."""
    from sr.minc.io import load_volume_np

    if args.verbose:
        print(f"Loading sigma from {args.sigma_file}")
    try:
        sig_np, _ = load_volume_np(str(args.sigma_file))
    except Exception as exc:
        sys.exit(f"ERROR loading sigma file {args.sigma_file}: {exc}")
    if sig_np.shape != shape:
        sys.exit(
            f"ERROR: sigma file shape {sig_np.shape} != input shape {shape}"
        )
    return sig_np


def _build_sigma_np(args: argparse.Namespace, vol_np, shape):
    """Resolve whichever sigma flag was given into a per-voxel numpy array."""
    import numpy as np

    if args.sigma_file is not None:
        return _load_sigma_file(args, shape).astype(np.float64)

    if args.sigma is not None:
        if args.verbose:
            print(f"Uniform sigma: {args.sigma}")
        return np.full_like(vol_np, float(args.sigma))

    if args.auto_sigma_voxel:
        from sr_numpy.core import _noise_level

        if args.verbose:
            print("Estimating per-voxel sigma from input volume ...")
        return _noise_level(vol_np)

    from sr_numpy.noise_estimate import noise_estimate

    if args.verbose:
        print("Estimating uniform sigma (noise_estimate) ...")
    level, mean_signal = noise_estimate(vol_np, gaussian=args.gauss)
    if args.verbose:
        print(f"  uniform sigma={level:.4f}  mean_signal={mean_signal:.4f}")
    return np.full_like(vol_np, float(level))


def _build_sigma(args: argparse.Namespace, vol_t, shape):
    """Resolve whichever sigma flag was given into a per-voxel tensor."""
    import torch

    if args.sigma_file is not None:
        return torch.from_numpy(_load_sigma_file(args, shape)).to(
            device=vol_t.device, dtype=vol_t.dtype)

    if args.sigma is not None:
        if args.verbose:
            print(f"Uniform sigma: {args.sigma}")
        return torch.full_like(vol_t, float(args.sigma))

    if args.auto_sigma_voxel:
        from sr_pytorch.core import _noise_level

        if args.verbose:
            print("Estimating per-voxel sigma from input volume ...")
        return _noise_level(vol_t)

    # --auto-sigma-global, and the default when no sigma flag is given.
    from sr_pytorch.noise_estimate import noise_estimate

    if args.verbose:
        print("Estimating uniform sigma (noise_estimate) ...")
    level, mean_signal = noise_estimate(vol_t.to(torch.float64),
                                        gaussian=args.gauss)
    if args.verbose:
        print(f"  uniform sigma={level:.4f}  mean_signal={mean_signal:.4f}")
    return torch.full_like(vol_t, float(level))


def _run_numpy(args, vol_np, factors, mean_preserve):
    """The experimental torch-free path: returns the filtered float64 array."""
    import numpy as np
    from sr_numpy.regularize import regularize

    sigma = _build_sigma_np(args, vol_np, vol_np.shape)
    if args.verbose:
        print(f"  sigma: min={sigma.min():.3f}  max={sigma.max():.3f}  "
              f"mean={sigma.mean():.3f}")
        print("Running NLM filtering (numpy) ...")
    out = regularize(vol_np, args.v, args.f, sigma, factors=factors,
                     mean_preserve=mean_preserve, strength=args.strength,
                     floor_mode=args.floor_mode)
    return np.asarray(out, dtype=np.float64)


def _run_pytorch(args, vol_np, factors, mean_preserve):
    """The default PyTorch path: returns the filtered float64 array."""
    import torch
    from sr_pytorch.regularize import regularize

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32 if device.type == "cuda" else torch.float64

    if args.verbose:
        print(f"Device  : {device}  dtype={dtype}")

    vol_t = torch.from_numpy(vol_np).to(device=device, dtype=dtype)
    sigma_t = _build_sigma(args, vol_t, vol_np.shape)

    if args.verbose:
        sig_np_report = sigma_t.detach().cpu().numpy()
        print(f"  sigma: min={sig_np_report.min():.3f}  "
              f"max={sig_np_report.max():.3f}  mean={sig_np_report.mean():.3f}")
        print("Running NLM filtering ...")

    out_t = regularize(vol_t, args.v, args.f, sigma_t, factors=factors,
                       mean_preserve=mean_preserve, strength=args.strength,
                       floor_mode=args.floor_mode)
    return out_t.cpu().to(torch.float64).numpy()


def main() -> None:
    args = _parse_args()

    try:
        from sr.minc.io import format_history, load_volume_np, save_volume
    except ImportError as exc:
        sys.exit(f"ERROR: Could not import I/O helpers: {exc}")

    factors = tuple(args.factors)
    mean_preserve = _resolve_mean_preserve(args.mean_preserve, factors)

    if args.verbose:
        print(f"Backend : {args.backend}")
        print(f"Params  : v={args.v}  f={args.f}  strength={args.strength}  "
              f"floor_mode={args.floor_mode}")
        print(f"          factors={factors}  mean_preserve={mean_preserve}")

    if args.verbose:
        print(f"Loading {args.input}")
    try:
        vol_np, v2w = load_volume_np(str(args.input))
    except Exception as exc:
        sys.exit(f"ERROR loading {args.input}: {exc}")

    if args.verbose:
        print(f"  shape={vol_np.shape}  min={vol_np.min():.1f}  max={vol_np.max():.1f}")

    runner = _run_numpy if args.backend == "numpy" else _run_pytorch
    start = time.time()
    out_np = runner(args, vol_np, factors, mean_preserve)
    elapsed = time.time() - start

    if args.verbose:
        print(f"  output: min={out_np.min():.1f}  max={out_np.max():.1f}")
        print(f"  elapsed: {elapsed:.2f}s")

    history = format_history(sys.argv)
    if args.verbose:
        print(f"Saving {args.output}")
    try:
        save_volume(
            str(args.output),
            out_np,
            v2w,
            ref_fname=str(args.input),
            history=history,
        )
    except Exception as exc:
        sys.exit(f"ERROR saving {args.output}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
