"""One-level forward 3-D Daubechies-4 DWT, NumPy port of :mod:`sr_pytorch.dwt`.

Experimental torch-free backend (Manjon 2010 denoising pipeline).  Behaviour
matches :mod:`sr_pytorch.dwt` (itself a port of EZminc's ``dwt_forward``);
that module on ``float64`` is the numerical reference.

The C++ ``pwt`` reads its four taps at ``(2k + j) mod n``.  The PyTorch port
reproduces this with a circular pad; here it is written directly as periodic
tap indexing, which is the same thing.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# D4 coefficients (``dwt.cpp``'s ``C0..C3``) and the high-pass filter
# ``cr`` (``C3, -C2, C1, -C0``).
_DEC_LO = np.array([-0.1294095225512604, 0.2241438680420134,
                    0.8365163037378079, 0.4829629131445341])
_DEC_HI = np.array([-0.4829629131445341, 0.8365163037378079,
                    -0.2241438680420134, -0.1294095225512604])


def _conv_along(t: np.ndarray, axis: int) -> np.ndarray:
    """One ``pwt`` step: split ``t`` along ``axis`` into low and high halves.

    Returns an array that leads with a 2 (low, high) and carries the halved
    axis at position ``axis + 1``, exactly as the PyTorch port lays it out.
    """
    t = np.asarray(t)
    n = t.shape[axis]
    m = n // 2
    idx = (2 * np.arange(m)[:, None] + np.arange(4)[None, :]) % n  # (m, 4)
    win = np.take(t, idx, axis=axis)            # m at `axis`, size-4 at axis+1
    # PyTorch correlates with the reversed filter (``filt.flip(0)``); the tap
    # at offset j therefore multiplies ``_DEC_*[3 - j]``.
    lo = np.tensordot(win, _DEC_LO[::-1], axes=([axis + 1], [0]))
    hi = np.tensordot(win, _DEC_HI[::-1], axes=([axis + 1], [0]))
    return np.stack([lo, hi], axis=0)           # halved axis now at axis + 1


def dwt3d_db4(x: np.ndarray) -> dict:
    """One level of the D4 transform on a ``(1, 1, P, P, P)`` array.

    Transforms z, then y, then x.  Returns the eight subbands keyed by their
    quadrant names; bit 0 of the index is the x half.
    """
    xd = _conv_along(x, 2)
    xdh = np.stack([_conv_along(xd[i], 3) for i in range(2)])
    xdhw = np.stack([_conv_along(xdh[i, j], 4)
                     for i in range(2) for j in range(2)])
    xdhw = xdhw.reshape(2, 2, 2, *xdhw.shape[2:])
    names = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
    return {n: xdhw[i // 4, (i // 2) % 2, i % 2]
            for i, n in enumerate(names)}


def _find_nearest_square_pow2(n: int) -> int:
    """The smallest power of two at or above ``n``, halved when ``n`` is
    exactly that power of two."""
    m = n
    p = 0
    while m != 0:
        m >>= 1
        p += 1
    if n == (1 << (p - 1)):
        p -= 1
    return 1 << p


def dwt_forward(volume: np.ndarray) -> list:
    """The one-level forward DWT of ``volume`` (a ``(z, y, x)`` array).

    Zero-padded, centred, onto a ``(P, P, P)`` grid with ``P`` the nearest
    power of two at or above the largest extent, transformed once, and cut
    back into the eight subbands, returned in quadrant order
    ``(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)``.  Each subband is
    ``ceil(dim / 2)`` along every axis.
    """
    volume = np.asarray(volume, dtype=np.float64)
    sz, sy, sx = volume.shape
    p = _find_nearest_square_pow2(max(sz, sy, sx))
    p0, p1, p2 = (p - sz) // 2, (p - sy) // 2, (p - sx) // 2
    buf = np.zeros((p, p, p), dtype=np.float64)
    buf[p0:p0 + sz, p1:p1 + sy, p2:p2 + sx] = volume
    subs = dwt3d_db4(buf.reshape(1, 1, p, p, p))
    q0, q1, q2 = p0 // 2, p1 // 2, p2 // 2
    osz = ((sz + 1) // 2, (sy + 1) // 2, (sx + 1) // 2)
    names = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
    return [subs[n][0, 0, q0:q0 + osz[0], q1:q1 + osz[1], q2:q2 + osz[2]]
            for n in names]
