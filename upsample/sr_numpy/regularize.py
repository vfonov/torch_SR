"""NLM regularization kernel, NumPy port of :mod:`sr_pytorch.regularize`.

Experimental torch-free backend for plain denoising (Manjon 2010): a single
pass of ``cMRegularizarNLM3D_V2``-style non-local means.  The algorithm,
constants and the ``strength`` / ``floor_mode`` denoising extensions are
carried over unchanged from :mod:`sr_pytorch.regularize`, which on CPU /
``float64`` is the numerical reference for this module.

Everything runs in ``float64``.  The autograd-only knobs of the PyTorch
version (``detach_weights``, ``no_grad``) have no meaning here and are
dropped.

See ``sr_pytorch/regularize.py`` for the full description of ``strength``
(multiplies sigma; ``0`` is the identity) and ``floor_mode`` (``"legacy"``
fixed absolute noise floor, ``"scale"`` affine-equivariant floor from the
volume's own dynamic range).
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)

#: The intensity scale :data:`SIGMA_FLOOR` is calibrated on, used only when
#: ``floor_mode="scale"``.
DEFAULT_SCALE = 256.0

#: A voxel whose ``sigma`` (after ``strength``) is below this, on
#: :data:`DEFAULT_SCALE`, is left untouched when ``floor_mode="scale"``.
SIGMA_FLOOR = 1.0

#: The centiles whose separation stands for the volume's dynamic range.
SCALE_QUANTILES = (0.01, 0.99)


def _scale(volume: np.ndarray) -> float:
    """The volume's dynamic range: the 1st-to-99th centile separation.

    Nearest-rank centiles via ``np.partition`` (1-indexed rank
    ``ceil(q * count)``), matching ``torch.kthvalue`` in the PyTorch port.
    """
    values = np.asarray(volume, dtype=np.float64).reshape(-1)
    count = values.size
    ranks = [max(1, math.ceil(q * count)) for q in SCALE_QUANTILES]
    low, high = (np.partition(values, r - 1)[r - 1] for r in ranks)
    return float(high - low)


def _box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Mean over every ``(k, k, k)`` window, no padding (valid ``avg_pool3d``
    with unit stride)."""
    if k == 1:
        return arr
    windows = sliding_window_view(arr, (k, k, k))
    return windows.mean(axis=(-3, -2, -1))


def _block_means_full(arr: np.ndarray,
                      factors: Tuple[int, int, int]) -> np.ndarray:
    """Non-overlapping ``factors``-sized block means, tiled back to
    ``arr``'s shape (``repeat_interleave`` of a strided ``avg_pool3d``)."""
    fz, fy, fx = factors
    d, h, w = arr.shape
    means = arr.reshape(d // fz, fz, h // fy, fy, w // fx, fx).mean(
        axis=(1, 3, 5))
    return np.repeat(np.repeat(np.repeat(means, fz, 0), fy, 1), fx, 2)


def regularize(
    volume: np.ndarray,
    v: int,
    f: int,
    sigma: np.ndarray,
    factors: Tuple[int, int, int],
    mean_preserve: bool = True,
    strength: float = 1.0,
    floor_mode: str = "legacy",
) -> np.ndarray:
    """NLM regularization -- NumPy implementation.

    Args:
        volume:         Input volume, shape ``(D, H, W)``.
        v:              Search-neighbourhood radius.
        f:              Patch half-size; patch covers ``(2f+1)**3`` voxels.
        sigma:          Per-voxel noise-level estimate, same shape as *volume*.
        factors:        Upsampling factors ``(fz, fy, fx)``.
        mean_preserve:  When ``True`` each ``(fz x fy x fx)`` output block is
                        shifted to match its input block mean.  ``False`` for
                        pure per-voxel NLM filtering.
        strength:       Multiplies *sigma* before use (``1.0`` is a no-op,
                        ``0`` disables smoothing).
        floor_mode:     ``"legacy"`` (default) or ``"scale"``.

    Returns:
        Regularized volume, ``float64``, same shape as *volume*.
    """
    if floor_mode not in ("legacy", "scale"):
        raise ValueError(
            "regularize: floor_mode must be 'legacy' or 'scale' (got %r)"
            % (floor_mode,))

    volume = np.asarray(volume, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if strength != 1.0:
        sigma = sigma * strength

    D, H, W = volume.shape
    patch_k = 2 * f + 1
    total_pad = v + f

    # ---- pre-pad volume once (reflect BC) --------------------------------
    vol_big = np.pad(volume, total_pad, mode="reflect")
    vol_ctr_f = vol_big[v:v + D + 2 * f,
                        v:v + H + 2 * f,
                        v:v + W + 2 * f]

    # ---- block means for mean-preservation ------------------------------
    block_means_full = (_block_means_full(volume, factors)
                        if mean_preserve else None)

    # ---- local means for fast mean-difference rejection ----------------
    # PyTorch reflect-pads by 1 then valid-pools: scipy's "mirror".
    local_mean = uniform_filter(volume, size=3, mode="mirror")
    lm_pad = np.pad(local_mean, v, mode="reflect")

    # ---- per-voxel noise^2 --------------------------------------------
    if floor_mode == "scale":
        scale = _scale(volume)
        if not scale > 0.0:
            raise ValueError(
                "regularize: floor_mode='scale' needs a volume with nonzero "
                "dynamic range (%g-%g centile spread was %g)"
                % (SCALE_QUANTILES[0], SCALE_QUANTILES[1], scale))
        floor = SIGMA_FLOOR * scale / DEFAULT_SCALE
        h_sq = np.maximum(sigma, floor) ** 2
        sigma_mask = sigma >= floor
    else:
        h_sq = np.maximum(sigma ** 2, 1e-10)
        sigma_mask = sigma >= 1.0

    # ---- NLM accumulation --------------------------------------------
    weighted_sum = np.zeros((D, H, W), dtype=np.float64)
    weight_total = np.ones((D, H, W), dtype=np.float64)

    for dz in range(-v, v + 1):
        for dy in range(-v, v + 1):
            for dx in range(-v, v + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue

                sz, sy, sx = v + dz, v + dy, v + dx

                shifted_f = vol_big[sz:sz + D + 2 * f,
                                    sy:sy + H + 2 * f,
                                    sx:sx + W + 2 * f]
                shifted = vol_big[sz + f:sz + f + D,
                                  sy + f:sy + f + H,
                                  sx + f:sx + f + W]

                diff_sq = (vol_ctr_f - shifted_f) ** 2
                patch_dist = _box_mean(diff_sq, patch_k)

                shifted_lm = lm_pad[v + dz:v + dz + D,
                                    v + dy:v + dy + H,
                                    v + dx:v + dx + W]

                exponent = np.maximum(patch_dist / (2.0 * h_sq) - 1.0, 0.0)
                w = np.exp(-exponent)
                w[np.abs(local_mean - shifted_lm) > 0.6 * sigma] = 0.0
                w[~sigma_mask] = 0.0

                weighted_sum += w * shifted
                weight_total += w

    output = (volume + weighted_sum) / weight_total

    # ---- mean-preservation correction -------------------------------
    if not mean_preserve:
        return output

    out_block_means_full = _block_means_full(output, factors)
    return output + (block_means_full - out_block_means_full)
