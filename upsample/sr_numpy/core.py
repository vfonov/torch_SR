"""Per-voxel noise-level helpers, NumPy port of :mod:`sr_pytorch.core`.

Experimental torch-free backend for plain denoising (Manjon 2010).  Only the
two noise-level helpers that ``upsample/denoise.py`` needs are ported here; the
super-resolution pipeline (``sr_pytorch.core.nlm_upsample``) is out of scope.
:mod:`sr_pytorch.core` on CPU / ``float64`` is the numerical reference.

Both helpers use ``scipy.ndimage.uniform_filter(size=3, mode="reflect")``,
which matches the reflect-padded ``avg_pool3d`` of the PyTorch version.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def _local_std(vol: np.ndarray) -> np.ndarray:
    """Population std in a 3x3x3 neighbourhood (reflect BC).  # (D, H, W)"""
    vol = np.asarray(vol, dtype=np.float64)
    lmean = uniform_filter(vol, size=3, mode="mirror")
    lsqmean = uniform_filter(vol * vol, size=3, mode="mirror")
    return np.sqrt(np.maximum(lsqmean - lmean * lmean, 0.0))


def _noise_level(vol: np.ndarray) -> np.ndarray:
    """Per-voxel noise level: smooth the local std, then halve.  # (D, H, W)"""
    sigma = _local_std(vol)
    smoothed = uniform_filter(sigma, size=3, mode="mirror")
    return smoothed / 2.0
