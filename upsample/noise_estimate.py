#!/usr/bin/env python3
"""Estimate the Rician (or Gaussian) noise level of a volume.

    python upsample/noise_estimate.py input.mnc [--noise] [--snr]

Input format is chosen by extension: ``.mnc`` (needs ``minc2_simple``) or
``.nii``/``.nii.gz`` (needs the optional ``nibabel`` package -- ``pip
install nibabel``). See ``sr.minc.io``.

Mirrors the EZminc ``noise_estimate`` tool (Coupe et al. 2009) and its
torch_n3 CLI (``torch_n3/noise_estimate_cli.py``): the same ``input``,
``--verbose``, ``--noise``, ``--snr``, ``--bins``, ``--mask`` and ``--gauss``
options, restructured to use this project's own ``sr.minc.io`` for volume I/O
instead of torch_n3's ``Volume`` class. Runs
``sr_pytorch.noise_estimate.noise_estimate`` alone; there is no
``--backend`` here, since there is exactly one implementation. Prints one
number per line: the noise level (the default), the signal-to-noise ratio
with ``--snr``, or both with ``--noise --snr``. Verbose diagnostics go to
stderr so the numeric output stays parseable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_UPSAMPLE = Path(__file__).resolve().parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upsample.noise_estimate",
        description="Estimate the Rician (or, with --gauss, Gaussian) "
                    "noise level of a volume (Coupe et al. 2009).")
    parser.add_argument("input", type=Path,
                        help="Volume to estimate noise for (.mnc, .nii, .nii.gz)")

    parser.add_argument("--verbose", action="store_true",
                        help="report the estimate and elapsed time on stderr")
    parser.add_argument("--noise", action="store_true",
                        help="print the noise level (the default)")
    parser.add_argument("--snr", action="store_true",
                        help="print the signal-to-noise ratio")
    parser.add_argument("--bins", type=int, default=2000,
                        help="histogram bins used to detect the object "
                             "(default: %(default)s)")
    parser.add_argument("--mask", type=Path, default=None,
                        help="object mask (.mnc, .nii, .nii.gz), overriding "
                             "the built-in k-means object detection")
    parser.add_argument("--gauss", action="store_true",
                        help="assume Gaussian rather than Rician noise")
    parser.add_argument("--device", default=None,
                        help="run on this torch device, e.g. cuda")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from sr.minc.io import load_volume_np
    except ImportError as exc:
        sys.exit(f"ERROR: Could not import I/O helpers: {exc}")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        vol_np, _ = load_volume_np(str(args.input))
    except Exception as exc:
        sys.exit(f"ERROR loading {args.input}: {exc}")
    volume = torch.from_numpy(vol_np).to(device=device, dtype=torch.float64)

    mask = None
    if args.mask is not None:
        try:
            mask_np, _ = load_volume_np(str(args.mask))
        except Exception as exc:
            sys.exit(f"ERROR loading mask {args.mask}: {exc}")
        mask = torch.from_numpy(mask_np).to(device=device)

    from sr_pytorch.noise_estimate import noise_estimate

    start = time.time()
    noise, mean_signal = noise_estimate(
        volume, mask=mask, gaussian=args.gauss, hist_bins=args.bins)
    elapsed = time.time() - start

    if args.verbose:
        print("noise=%g signal=%g snr=%g" % (
            noise, mean_signal, mean_signal / noise), file=sys.stderr)
        print("estimated in %.2fs (device=%s)" % (
            elapsed, args.device or "cpu"), file=sys.stderr)

    if args.noise or not args.snr:
        print("%g" % noise)
    if args.snr:
        print("%g" % (mean_signal / noise))

    return 0


if __name__ == "__main__":
    sys.exit(main())
