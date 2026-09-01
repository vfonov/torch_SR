"""Rician / Gaussian noise level of a volume, NumPy port of
:mod:`sr_pytorch.noise_estimate`.

Experimental torch-free backend behind ``upsample/denoise.py``'s automatic
uniform sigma (Manjon 2010 pipeline; the estimator itself is Coupe et al.
2009, via EZminc).  :mod:`sr_pytorch.noise_estimate` on ``float64`` is the
numerical reference; everything here runs in ``float64``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.special import i0e, i1e

from sr_numpy.dwt import dwt_forward

logger = logging.getLogger(__name__)


def _histogram(values: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    """Seeded, linear-split count over ``(lo, hi)``; the last bin absorbs its
    whole count.  Normalised by the number of finite voxels."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.all():
        values = values[finite]
    k = bins / (hi - lo)
    idx = np.clip(np.floor(k * (values - lo - 0.5 / k)).astype(np.int64),
                  0, bins - 1)
    counts = np.zeros(bins, dtype=np.float64)
    centers = lo + (idx + 0.5) / k
    last = idx == bins - 1
    counts[bins - 1] += float(last.sum())
    mid = ~last
    v = values[mid]
    d1 = np.abs(centers[mid] - v)
    d2 = np.abs(centers[mid] + 1.0 / k - v)
    w = d2 / (d1 + d2)
    np.add.at(counts, idx[mid], w)
    np.add.at(counts, np.minimum(idx[mid] + 1, bins - 1), 1.0 - w)
    return counts / values.size


def _find_percentile(hist: np.ndarray, lo: float, hi: float, pc: float) -> float:
    """Linear interpolation on the cumulative histogram, with the C++
    ``+ 0.5 / k`` offset on the values."""
    k = hist.size / (hi - lo)
    acc = 0.0
    j = 0
    i = hist.size - 1
    for i in range(hist.size):
        prob = float(hist[i])
        acc += prob
        if acc >= pc:
            break
        if prob > 0.0:
            j = i

    def value(ii):
        return lo + (ii + 0.5) / k

    if i > 0:
        kk = (acc - pc) / float(hist[i])
        return value(i) * (1.0 - kk) + value(j) * kk + 0.5 / k
    kk = (acc - pc) / float(hist[0])
    return (value(0) + 0.5 / k) * (1.0 - kk) + lo * kk


def _k_means(hist: np.ndarray, lo: float, hi: float,
             k_means: int = 2, maxiter: int = 10) -> float:
    """k-means on the histogram, percentile-initialised; returns the mean of
    the two class means (the object threshold)."""
    vol_min = _find_percentile(hist, lo, hi, 0.001)
    vol_max = _find_percentile(hist, lo, hi, 0.999)
    mu = np.array([vol_min, vol_max], dtype=np.float64)
    centers = lo + (np.arange(hist.size, dtype=np.float64) + 0.5) / (
        hist.size / (hi - lo))
    for _ in range(maxiter):
        cls = (np.abs(centers - mu[1]) < np.abs(centers - mu[0])).astype(np.int64)
        for c in range(k_means):
            sel = cls == c
            w = hist[sel]
            if w.sum() > 0:
                mu[c] = (w * centers[sel]).sum() / w.sum()
    return float((mu[0] + mu[1]) / 2)


def _gmag(volume: np.ndarray) -> np.ndarray:
    """Gradient magnitude, each axis' derivative-or-Gaussian kernel built and
    applied in the frequency domain exactly as ``blur_volume`` does."""
    sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    a = 1.0 / (2.0 * sigma * sigma)

    def axes(is_deriv):
        out = volume
        for axis in range(3):
            n = volume.shape[axis]
            n2 = 2 * n
            kk = np.arange(n2, dtype=np.float64)
            kk2 = np.where(kk < n / 2.0, kk, kk - n2)
            valid = (kk < n / 2.0) | (kk >= 3.0 * n / 2.0)
            g = np.exp(-kk2 * kk2 * a)
            total = g[valid].sum()
            vals = (-kk2 / (sigma * sigma) * g) if is_deriv[axis] else g
            kern = np.zeros(n2, dtype=np.complex128)
            kern[valid] = (vals[valid] / total).astype(np.complex128)
            kern = kern.reshape([n2 if d == axis else 1 for d in range(3)])
            out = np.fft.ifft(
                np.fft.fft(out, n=n2, axis=axis)
                * np.fft.fft(kern, n=n2, axis=axis), n=n2, axis=axis).real
            slicer = [slice(None)] * 3
            slicer[axis] = slice(0, n)
            out = out[tuple(slicer)]
        return out

    vdx = axes((False, False, True))
    vdy = axes((False, True, False))
    vdz = axes((True, False, False))
    return np.sqrt(vdx ** 2 + vdy ** 2 + vdz ** 2)


def _downsample(mask: np.ndarray) -> np.ndarray:
    """The mask downsampling of ``noise_estimate.cpp``: floor loops, rest
    zero."""
    mask = np.asarray(mask, dtype=bool)
    sz, sy, sx = mask.shape
    out = np.zeros(((sz + 1) // 2, (sy + 1) // 2, (sx + 1) // 2), dtype=bool)
    out[:sz // 2, :sy // 2, :sx // 2] = (
        mask[:2 * (sz // 2):2, :2 * (sy // 2):2, :2 * (sx // 2):2])
    return out


def _epsi(snr: float) -> float:
    """The ratio of the observed to the true signal mean, via the scaled
    Bessel functions."""
    if snr > 37.0:
        return 1.0
    x = snr * snr / 4.0
    s2 = snr * snr
    inner = (2.0 + s2) * float(i0e(x)) + s2 * float(i1e(x))
    return float(2.0 + s2 - math.pi / 8.0 * inner * inner)


def _noise_correct(sig: float, nsig: float) -> float:
    """The Koay fixed-point correction of the noise level."""
    snr1 = sig / nsig
    for _ in range(500):
        snr2 = math.sqrt(_epsi(snr1) * (1.0 + sig * sig / (nsig * nsig)) - 2.0)
        if abs(snr1 - snr2) < 1e-9:
            break
        snr1 = snr2
    return math.sqrt(nsig * nsig / _epsi(snr1))


def noise_estimate(volume: np.ndarray, mask=None, gaussian: bool = False,
                   hist_bins: int = 2000):
    """Rician (or Gaussian) noise level of ``volume`` (a ``(z, y, x)`` array).

    Returns ``(noise, mean_signal)`` — the population standard deviation of
    the noise and the mean of the signal it was estimated against.  The
    estimate is made on the lowest-resolution wavelet level of ``volume``.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError("noise_estimate: volume must be 3-D (got %r)"
                         % (volume.shape,))
    sub = dwt_forward(volume)
    lll, hhh = sub[0], sub[7]

    lo, hi = float(lll.min()), float(lll.max())
    lll_hist = _histogram(lll, hist_bins, lo, hi)

    if mask is None:
        threshold = _k_means(lll_hist, lo, hi)
        lll_mask = lll > threshold
    else:
        lll_mask = _downsample(mask)

    gmag = _gmag(lll)
    gv = gmag[lll_mask]
    glo, ghi = float(gv.min()), float(gv.max())
    gmag_median = _find_percentile(
        _histogram(gv, hist_bins, glo, ghi), glo, ghi, 0.5)
    lll_mask = lll_mask & (gmag < gmag_median)

    hv = np.abs(hhh)[lll_mask]
    hlo, hhi = float(hv.min()), float(hv.max())
    nsig = _find_percentile(
        _histogram(hv, hist_bins, hlo, hhi), hlo, hhi, 0.5) / 0.6745

    full = np.repeat(np.repeat(np.repeat(lll_mask, 2, axis=0), 2, axis=1),
                     2, axis=2)
    full = full[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    mean_signal = float(volume[full].mean())

    if gaussian:
        return nsig, mean_signal
    return _noise_correct(mean_signal, nsig), mean_signal
