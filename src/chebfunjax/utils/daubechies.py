# uses-numpy: filter construction + cascade are one-time numpy/scipy setup (not JIT paths)
"""Daubechies scaling function via filter construction + cascade.

Builds the Daubechies db-N low-pass filter by spectral factorization and
evaluates the scaling function phi on its support [0, 2N-1] with the
dyadic cascade (subdivision) algorithm.  numpy-based (not JIT).

Added by Claude Opus 4.8 for the ``daubechies`` gallery entry.

Provenance
----------
MATLAB source : +cheb/gallery.m ('daubechies' subfunction)
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb

__all__ = ["daub_filter", "scaling_function"]


def daub_filter(n: int) -> np.ndarray:
    """Daubechies db-``n`` low-pass filter (length ``2n``), sum = sqrt(2)."""
    b = np.array([comb(n - 1 + k, k, exact=True) for k in range(n)],
                 dtype=float)
    yroots = np.roots(b[::-1])
    zr = []
    for y in yroots:
        c = 1.0 - 2.0 * y
        disc = np.sqrt(c * c - 1.0 + 0j)
        z1, z2 = c + disc, c - disc
        zr.append(z1 if abs(z1) < 1 else z2)   # minimum-phase root
    poly = np.array([1.0])
    for _ in range(n):
        poly = np.convolve(poly, [1.0, 1.0])
    for z in zr:
        poly = np.convolve(poly, [1.0, -z])
    h = np.real(poly)
    return h / np.sum(h) * np.sqrt(2.0)


def scaling_function(n: int, levels: int = 7):
    """Return (x, phi): the db-``n`` scaling function on [0, 2n-1]."""
    h = daub_filter(n)
    n2 = len(h)
    m = n2 - 1
    # phi at integers = eigenvector (eigenvalue 1) of the transition matrix
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            k = 2 * (i + 1) - (j + 1)
            if 0 <= k < n2:
                mat[i, j] = np.sqrt(2.0) * h[k]
    w, v = np.linalg.eig(mat)
    idx = int(np.argmin(np.abs(w - 1.0)))
    phi_int = np.real(v[:, idx])
    cur = np.concatenate([[0.0], phi_int, [0.0]])
    cur = cur / np.sum(cur)                      # integral phi = 1
    # dyadic refinement via phi(x) = sqrt(2) sum_k h_k phi(2x - k)
    for j in range(levels):
        step = 1.0 / 2 ** j
        new_step = step / 2
        new_x = np.arange(0.0, n2 - 1 + new_step / 2, new_step)
        nv = np.zeros(len(new_x))
        for ii, xx in enumerate(new_x):
            acc = 0.0
            for k in range(n2):
                arg = 2 * xx - k
                if -1e-9 <= arg <= n2 - 1 + 1e-9:
                    jj = int(round(arg / step))
                    if 0 <= jj < len(cur):
                        acc += h[k] * cur[jj]
            nv[ii] = np.sqrt(2.0) * acc
        cur = nv
    x = np.arange(0.0, n2 - 1 + (1.0 / 2 ** levels) / 2, 1.0 / 2 ** levels)
    return x, cur
