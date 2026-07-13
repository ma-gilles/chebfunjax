# uses-numpy: Toeplitz/null-space manipulation is one-shot numpy
"""Trigonometric (Fourier) rational approximation: trigpade.

Added by Claude Fable 5 (Big-Three directive, trig rational
approximation).

Provenance
----------
MATLAB source : @chebfun/trigpade.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space, toeplitz

__all__ = ["trigpade"]


def _trig_coeffs_ascending(f) -> np.ndarray:
    """MATLAB trigcoeffs layout [c_{-N} ... c_0 ... c_N] (odd length)
    from a single-piece trig chebfun."""
    c = np.asarray(f.funs[0].tech.coeffs)  # descending wavenumber
    c = c[::-1].astype(complex)            # ascending
    if len(c) % 2 == 0:
        # split the top mode symmetrically to make the length odd
        c = np.concatenate([[c[0] / 2.0], c[1:], [c[0] / 2.0]])
    return c


def _trig_chebfun_from_ascending(c, domain):
    """Build a trig chebfun from ascending Laurent coefficients."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, _Piece
    from chebfunjax.tech.trigtech import Trigtech
    c = np.atleast_1d(np.asarray(c, dtype=complex))
    tech = Trigtech.from_coeffs(jnp.asarray(c[::-1]))
    piece = _Piece(tech=tech,
                   interval=(float(domain[0]), float(domain[1])))
    return Chebfun(funs=[piece],
                   domain=Domain((float(domain[0]),
                                  float(domain[1]))))


def _laurent_approx(c, m, n, N, tol):
    """One-sided Laurent-Pade coefficients (MATLAB laurent_approx)."""
    col = c[(m + 1) + N: (m + n) + N + 1]
    row = c[np.arange((m + 1) + N, (m - n) + N, -1)]
    C = toeplitz(col, row)
    b = null_space(C)
    if b.size == 0:
        # fall back to the smallest right singular vector
        _, _, Vh = np.linalg.svd(C)
        b = Vh[-1, :].conj()[:, None]
    if b.shape[1] > 1:
        b = b[:, :1]
    b = b[:, 0]
    if abs(b[0]) < tol:
        raise ValueError(
            "trigpade: denominator zero at the origin detected")
    b = b / b[0]
    M = max(m, n)
    col2 = c[N: M + N + 1].copy()
    col2[0] = col2[0] / 2.0
    C2 = np.tril(toeplitz(col2, col2))
    bb = np.concatenate([b, np.zeros(M + 1 - len(b))])
    a = C2 @ bb
    return a, b


def _laurent_pade(c, m, n, tol):
    c = np.asarray(c, dtype=complex).ravel()
    N = (len(c) - 1) // 2
    if n == 0:
        ap = c[N: N + m + 1].copy()
        am = c[N:: -1][: m + 1].copy()
        ap[0] /= 2.0
        am[0] /= 2.0
        ap = np.concatenate([np.zeros(len(ap) - 1), ap])
        am = np.concatenate([np.zeros(len(am) - 1), am])
        return ap, np.array([1.0]), am[::-1], np.array([1.0])
    ap, bp = _laurent_approx(c, m, n, N, tol)
    c_rev = c[::-1]
    if np.max(np.abs(c - np.conj(c_rev))) < 10 * tol:
        am, bm = np.conj(ap), np.conj(bp)
    else:
        am, bm = _laurent_approx(c_rev, m, n, N, tol)
    ap = np.concatenate([np.zeros(len(ap) - 1), ap])
    bp = np.concatenate([np.zeros(len(bp) - 1), bp])
    am = np.concatenate([np.zeros(len(am) - 1), am])[::-1]
    bm = np.concatenate([np.zeros(len(bm) - 1), bm])[::-1]
    return ap, bp, am, bm


def _chop(c, tol):
    mid = (len(c) - 1) // 2
    nz = np.where(np.abs(c) > tol)[0]
    if len(nz) == 0:
        return np.zeros(1, dtype=c.dtype)
    nn = max(mid - nz[0], nz[-1] - mid)
    return c[mid - nn: mid + nn + 1]


def _center_pad(v, L):
    k = (len(v) - 1) // 2
    pad = L - k
    return np.concatenate([np.zeros(pad), v, np.zeros(pad)])


def trigpade(f, m: int, n: int):
    """Trigonometric (Fourier) Pade approximation of a periodic
    chebfun (MATLAB trigpade): returns
    ``(p, q, r, tn_p, td_p, tn_m, td_m)`` with
    ``p/q = tn_p/td_p + tn_m/td_m``.

    Provenance
    ----------
    MATLAB source : @chebfun/trigpade.m
    Chebfun commit: 7574c77
    """
    dom = (float(f.domain.a), float(f.domain.b))
    if getattr(f, "isempty", lambda: False)():
        return f
    c = _trig_coeffs_ascending(f)
    N = (len(c) - 1) // 2
    tol = 100 * np.finfo(float).eps * np.max(np.abs(c))

    d = 2 * max(m, n) - N
    if d > 0:
        c = np.concatenate([np.zeros(d), c, np.zeros(d)])
        N += d

    ap, bp, am, bm = _laurent_pade(c, m, n, tol)

    tn_p = _trig_chebfun_from_ascending(ap, dom)
    td_p = _trig_chebfun_from_ascending(bp, dom)
    tn_m = _trig_chebfun_from_ascending(am, dom)
    td_m = _trig_chebfun_from_ascending(bm, dom)

    L = (max(len(ap), len(am), len(bp), len(bm)) - 1) // 2
    ap_, bp_, am_, bm_ = (
        _center_pad(v, L) for v in (ap, bp, am, bm))
    pk = np.convolve(ap_, bm_) + np.convolve(am_, bp_)
    qk = np.convolve(bm_, bp_)
    pk = _chop(pk, tol)
    qk = _chop(qk, tol)

    p = _trig_chebfun_from_ascending(pk, dom)
    q = _trig_chebfun_from_ascending(qk, dom)

    # discard imaginary rounding errors for real input
    if np.max(np.abs(c - np.conj(c[::-1]))) < tol:
        p = p.real()
        q = q.real()

    def r(t):
        return p(t) / q(t)

    return p, q, r, tn_p, td_p, tn_m, td_m
