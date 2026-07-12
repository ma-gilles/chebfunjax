# uses-numpy: discrete transforms are one-shot numpy/scipy operations
"""Discrete cosine/sine/Legendre transforms (MATLAB chebfun.dct family).

Added by Claude Fable 5 (MISSING_FEATURES named-utilities sweep).

Provenance
----------
MATLAB source : chebfun.dct / chebfun.dst / chebfun.dlt / chebfun.idlt
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct as _sdct
from scipy.fft import dst as _sdst

__all__ = ["dct", "idct", "dst", "idst", "dlt", "idlt"]


def dct(x, kind: int = 2):
    """Discrete cosine transform, MATLAB chebfun.dct normalization
    (plain unnormalized sums, types I-IV)."""
    x = np.asarray(x, dtype=float)
    return _sdct(x, type=kind, axis=0, norm=None) / 2.0 \
        if kind in (2, 3) else _sdct(x, type=kind, axis=0, norm=None) / 2.0


def idct(x, kind: int = 2):
    """Inverse DCT (the inverse of :func:`dct` of the same kind)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    inv_kind = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
    scale = {1: 2.0 / max(n - 1, 1), 2: 2.0 / n, 3: 2.0 / n,
             4: 2.0 / n}[kind]
    return _sdct(x, type=inv_kind, axis=0, norm=None) * scale / 2.0


def dst(x, kind: int = 1):
    """Discrete sine transform (unnormalized sums)."""
    x = np.asarray(x, dtype=float)
    return _sdst(x, type=kind, axis=0, norm=None) / 2.0


def idst(x, kind: int = 1):
    """Inverse DST of the same kind."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    inv_kind = {1: 1, 2: 3, 3: 2, 4: 4}[kind]
    scale = {1: 2.0 / (n + 1), 2: 2.0 / n, 3: 2.0 / n,
             4: 2.0 / n}[kind]
    return _sdst(x, type=inv_kind, axis=0, norm=None) * scale / 2.0


def _legendre_vandermonde(x, n):
    P = np.zeros((len(x), n))
    P[:, 0] = 1.0
    if n > 1:
        P[:, 1] = x
    for k in range(1, n - 1):
        P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) \
            / (k + 1)
    return P


def dlt(c):
    """Discrete Legendre transform: values of the Legendre series with
    coefficients c at the Gauss-Legendre points (MATLAB chebfun.dlt)."""
    from chebfunjax.utils.quadrature import legpts
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    x, _ = (np.asarray(v) for v in legpts(n))
    return _legendre_vandermonde(x, n) @ c


def idlt(v):
    """Inverse DLT: Legendre coefficients from values at Gauss-Legendre
    points, via Gauss quadrature orthogonality (MATLAB chebfun.idlt)."""
    from chebfunjax.utils.quadrature import legpts
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    x, w = (np.asarray(t) for t in legpts(n))
    P = _legendre_vandermonde(x, n)
    k = np.arange(n)
    return (k + 0.5) * (P.T @ (w * v))
