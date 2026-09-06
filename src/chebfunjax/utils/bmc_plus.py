"""Compression-based addition for BMC-structured low-rank functions
(MATLAB ``@separableApprox/plus.m`` ``compression_plus``, used by
``@spherefun/plus.m`` and ``@diskfun/plus.m`` per parity block; Fable 5).

Provenance
----------
MATLAB source : @separableApprox/plus.m (compression_plus),
    @spherefun/plus.m, @diskfun/plus.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np  # uses-numpy: host-side dense QR/SVD

_EPS = float(np.finfo(np.float64).eps)


def _trig_grid(n):
    return -1.0 + 2.0 * np.arange(n) / n


def _trig_values(techs, nq):
    """Values of Trigtech slices on the nq-point equispaced grid."""
    from chebfunjax.tech.trigtech import _trig_eval_np
    x = _trig_grid(nq)
    cols = []
    for t in techs:
        v = np.asarray(_trig_eval_np(np.asarray(t.coeffs)[:, None], x,
                                     is_real=t.is_real)).ravel()
        cols.append(np.real(v) if t.is_real else v)
    return np.column_stack(cols), np.full(nq, 2.0 / nq)


def _cheb_values(techs, nq):
    """Values of Chebtech2 slices on the nq-point 2nd-kind grid with the
    Clenshaw-Curtis weights."""
    from chebfunjax.utils.quadrature import chebpts, chebweights
    x = np.asarray(chebpts(nq, kind=2))
    w = np.asarray(chebweights(nq, kind=2))
    cols = [np.asarray(t(jnp.asarray(x))).ravel() for t in techs]
    return np.column_stack(cols), w


def _n_coeffs(techs):
    return max(int(np.asarray(t.coeffs).shape[0]) for t in techs)


def compress_block(cols, rows, pivots, col_kind: str, vscl: float):
    """MATLAB compression_plus on a concatenated CDR block.

    ``cols``/``rows`` are lists of techs (already concatenated from the
    two operands), ``pivots`` the corresponding pivot values (the CDR
    diagonal is ``1 / pivots``).  ``col_kind`` is ``'trig'`` (spherefun
    columns) or ``'cheb'`` (diskfun columns); rows are always trig.
    Returns ``(col_vals, xc_kind, row_vals, new_pivots)`` where the value
    matrices are on the quadrature grids used (trig equispaced /
    Chebyshev 2nd kind) so the caller can rebuild techs.
    """
    r = len(cols)
    if r == 0:
        return None
    nqc = 2 * _n_coeffs(cols) + 3
    nqr = 2 * _n_coeffs(rows) + 2
    if col_kind == "trig":
        nqc = 2 * _n_coeffs(cols) + 2
        C, wc = _trig_values(cols, nqc)
    else:
        C, wc = _cheb_values(cols, nqc)
    R, wr = _trig_values(rows, nqr)
    d = np.asarray(pivots, dtype=float)
    dinv = np.where(np.abs(d) > 0, 1.0 / np.where(d == 0, 1.0, d), 0.0)
    # Fold the CDR diagonal into the columns before the QR: a GE column
    # has magnitude |pivot| (down to ~1e-14 for the last terms of a
    # derivative), and a QR of columns of such disparate scales loses
    # the small ones to rounding of the large ones (observed 1e-8 in
    # ``2f - f - f``).  Scaled, every column is O(1).
    Qc, Rc = np.linalg.qr(np.sqrt(wc)[:, None] * (C * dinv[None, :]))
    Qr, Rr = np.linalg.qr(np.sqrt(wr)[:, None] * R)
    U, s, Vt = np.linalg.svd(Rc @ Rr.T)
    keep = np.nonzero(s > 10 * _EPS * vscl)[0]
    if keep.size == 0:
        return None
    idx = int(keep[-1]) + 1            # MATLAB: find(..., 1, 'last')
    newC = (Qc @ U[:, :idx]) / np.sqrt(wc)[:, None]
    newR = (Qr @ np.conj(Vt[:idx, :]).T) / np.sqrt(wr)[:, None]
    return newC, newR, 1.0 / s[:idx]


def block_vscale(cols, rows, pivots, col_kind: str) -> float:
    """max |f| of a CDR block on its quadrature grids."""
    if len(cols) == 0:
        return 0.0
    if col_kind == "trig":
        C, _ = _trig_values(cols, 2 * _n_coeffs(cols) + 2)
    else:
        C, _ = _cheb_values(cols, 2 * _n_coeffs(cols) + 3)
    R, _ = _trig_values(rows, 2 * _n_coeffs(rows) + 2)
    d = np.asarray(pivots, dtype=float)
    D = np.where(np.abs(d) > 0, 1.0 / np.where(d == 0, 1.0, d), 0.0)
    return float(np.max(np.abs((C * D) @ R.T)))


def techs_from_values(vals, kind: str, keep: int | None = None):
    """Rebuild techs (one per column of ``vals``) from quadrature-grid
    values.  With ``keep`` the slice is resampled on the ``keep``-point
    grid of its kind: the sum of two CDR blocks lies in the span of the
    original slices, so the longer operand's length represents it
    exactly (MATLAB's quasimatrix QR keeps that length) and the extra
    quadrature modes are rounding."""
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.tech.trigtech import Trigtech, _trig_eval_np
    from chebfunjax.utils.quadrature import chebpts
    out = []
    for j in range(vals.shape[1]):
        col = vals[:, j]
        if kind == "trig":
            v = jnp.asarray(col if np.iscomplexobj(vals) else np.real(col))
            t = Trigtech.from_values(v)
            if keep is not None and v.shape[0] > keep:
                xs = _trig_grid(int(keep))
                w = np.asarray(_trig_eval_np(np.asarray(t.coeffs)[:, None], xs,
                                             is_real=t.is_real)).ravel()
                w = np.real(w) if t.is_real else w
                t = Trigtech.from_values(jnp.asarray(w))
            out.append(t)
        else:
            v = jnp.asarray(np.real(col))
            t = Chebtech2.from_values(v)
            if keep is not None and v.shape[0] > keep:
                xs = jnp.asarray(chebpts(int(keep), kind=2))
                t = Chebtech2.from_values(jnp.asarray(np.asarray(t(xs)).ravel()))
            out.append(t)
    return out
