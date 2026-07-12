"""Trigonometric polynomial utilities.

Translated from MATLAB Chebfun (commit 7574c77): trigpoly.m, diffbarytrig.m.
Original: Copyright 2017-2018 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ===========================================================================
# Trigonometric polynomial on [-1, 1]
# ===========================================================================


def trigpoly(
    n: int | jnp.ndarray,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> jnp.ndarray:
    """Evaluate a trigonometric polynomial exp(i*pi*n*x) on a standard grid.

    TRIGPOLY(N) returns the degree-N trigonometric polynomial exp(i*pi*N*x)
    evaluated at M = 2*|N|+1 equispaced points on [-1, 1).  N may be a
    vector of integers, in which case the result has one column per entry.

    The polynomial has period 2 (mapped from 2*pi).

    Parameters
    ----------
    n : int or array_like of ints
        Degree(s).  Must be integers.
    domain : (a, b), default (-1, 1)
        Interval.  The polynomial has period b-a.

    Returns
    -------
    vals : jnp.ndarray, shape (M,) or (M, len(n))
        Values of exp(i*pi*n*(x-a)*2/(b-a)) at M equispaced points.

    Notes
    -----
    In MATLAB Chebfun, ``trigpoly(N)`` returns a chebfun object; here we
    return the array of Fourier coefficients (a length-2N+1 zero vector with
    a single 1 at position N) and the equispaced evaluation grid.

    Provenance
    ----------
    MATLAB source : trigpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    diffbarytrig
    """
    # Check the original input for integer-ness before any truncating cast
    n_float = np.asarray(n, dtype=np.float64).ravel()
    if not np.all(n_float == np.round(n_float)):
        raise ValueError("n must contain integers.")
    n_arr = jnp.asarray(np.round(n_float).astype(np.int64)).ravel()

    a, b = float(domain[0]), float(domain[1])
    L = b - a

    N_max = int(jnp.max(jnp.abs(n_arr)))
    M = 2 * N_max + 1  # number of evaluation points

    # Equispaced points on [a, b)
    x = jnp.linspace(a, b, M + 1, dtype=jnp.float64)[:-1]

    # Angular frequency: exp(i * 2*pi/L * k * (x - a))
    # For each degree k in n_arr, this is cos(2pi/L*k*x) + i*sin(...)
    results = []
    for k in np.array(n_arr):
        freq = 2.0 * jnp.pi / L * float(k)
        vals = jnp.exp(1j * freq * (x - a))
        results.append(vals)

    if len(results) == 1:
        return results[0]
    return jnp.stack(results, axis=1)


# ===========================================================================
# Trigonometric barycentric differentiation
# ===========================================================================

# uses-numpy: iterative derivative polynomial computation

def diffbarytrig(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    N: int = 1,
    form: str = "odd",
) -> jnp.ndarray:
    """Derivative of a trigonometric rational function in barycentric form.

    D = DIFFBARYTRIG(ZZ, ZJ, FJ, WJ) returns the first derivative of the
    odd trigonometric barycentric rational function with support points ZJ,
    function values FJ, and barycentric weights WJ, evaluated at ZZ.

    D = DIFFBARYTRIG(ZZ, ZJ, FJ, WJ, N) computes the N-th derivative.

    D = DIFFBARYTRIG(ZZ, ZJ, FJ, WJ, N, FORM) uses the basis FORM, where
    FORM is 'odd' (default, uses csc) or 'even' (uses cot).

    Parameters
    ----------
    zz : jnp.ndarray
        Evaluation points (any shape).
    zj : jnp.ndarray, shape (m,)
        Support points.
    fj : jnp.ndarray, shape (m,)
        Function values at support points.
    wj : jnp.ndarray, shape (m,)
        Barycentric weights.
    N : int, default 1
        Order of derivative.
    form : {'odd', 'even'}, default 'odd'
        Barycentric basis type.

    Returns
    -------
    d : jnp.ndarray
        Derivative values at ZZ (same shape as ZZ).

    Notes
    -----
    The derivative is computed using the formula from Baddoo 2021 [1], which
    extends the standard barycentric differentiation formula to periodic
    rational functions.

    References
    ----------
    .. [1] P. J. Baddoo, "The AAAtrig algorithm for rational approximation
       of periodic functions", SIAM J. Sci. Comp. (2021).

    Provenance
    ----------
    MATLAB source : diffbarytrig.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2018 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    trigpoly
    """
    if N == 0:
        raise ValueError("N=0: use revaltrig for function evaluation, not diffbarytrig.")

    # Work in numpy for the iterative derivative computation
    zz_np = np.array(zz, dtype=complex).ravel()
    zj_np = np.array(zj, dtype=complex)
    fj_np = np.array(fj, dtype=complex)
    wj_np = np.array(wj, dtype=complex)

    npts = len(zz_np)
    m = len(zj_np)

    # Project onto first period window
    zvp = zz_np - 2 * np.pi * np.floor(np.real(zz_np / (2 * np.pi)))

    # Basis functions
    if form == "even":
        def cst(z):
            return 1.0 / np.tan(z)  # cot
    else:
        def cst(z):
            return 1.0 / np.sin(z)  # csc

    # Cauchy matrix: CC[i,j] = cst((zvp[i] - zj[j])/2)
    diff_half = (zvp[:, None] - zj_np[None, :]) / 2.0
    CC = cst(diff_half)  # (npts, m)

    rpDen = CC @ wj_np  # denominator
    rn = (CC @ (wj_np * fj_np)) / rpDen  # function values

    # Store results for each derivative order p=0..N
    rp = np.zeros((npts, N + 1), dtype=complex)
    rp[:, 0] = rn

    # Differentiation matrix (m x m) for higher-order derivatives
    D_list = [np.eye(m, dtype=complex)]  # D_list[p] = D^{(p)}

    for p in range(1, N + 1):
        # Derivative of rp[:, p-1] away from support points
        DR = np.zeros((npts, m, p), dtype=complex)
        for q in range(p):
            # Binomial coefficient C(p, q)
            binom = _binom(p, q)
            dcst_vals = _diff_cst((zvp[:, None] - zj_np[None, :]) / 2.0, p - q, form)
            fj_minus_rp = (fj_np[None, :] if q == 0 else 0.0) - rp[:, q:q + 1]
            DR[:, :, q] = (
                binom * (2.0 ** (q - p)) * dcst_vals * fj_minus_rp * wj_np[None, :]
            )

        rp[:, p] = np.sum(DR, axis=(1, 2)) / rpDen

        # Differentiation matrix D^{(p)} (MATLAB diffbarytrig.m lines 69-83)
        D_new = np.zeros((m, m), dtype=complex)
        diff_zj = (zj_np[:, None] - zj_np[None, :]) / 2.0  # (m, m)
        for q in range(1, p + 1):
            binom = _binom(p, q)
            Dp_q = D_list[p - q]  # (m, m)
            # MATLAB firstSum evaluates diffCstInv at ZERO ((zj - zj)/2),
            # a scalar per q — not at the pairwise differences — and the
            # weight ratio is wj.'./wj, i.e. entry (i,j) = wj[j]/wj[i].
            dcst_inv_zero = complex(
                np.asarray(_diff_cst_inv(np.zeros((1, 1)), q, form))[0, 0]
            )
            diag_Dpq = np.diag(Dp_q)  # (m,)
            first_sum = (
                (wj_np[None, :] / wj_np[:, None])
                * diag_Dpq[:, None]
                * (2.0 ** (-q))
                * dcst_inv_zero
            )
            second_sum = Dp_q * (2.0 ** (-q)) * _diff_cst_inv(diff_zj, q, form)

            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = cst(diff_zj) * binom * (first_sum - second_sum)
            D_new += np.where(np.isfinite(contrib), contrib, 0.0)

        np.fill_diagonal(D_new, 0.0)
        D_new -= np.diag(np.sum(D_new, axis=1))
        D_list.append(D_new)

    d_np = rp[:, N].copy()

    # Fix NaN at support points (0/0 case)
    DZJ = D_list[N] @ fj_np
    for ii in np.where(np.isnan(d_np))[0]:
        if not np.isnan(zvp[ii]):
            matches = np.where(zvp[ii] == zj_np)[0]
            if len(matches) > 0:
                d_np[ii] = DZJ[matches[0]]

    # Values at imaginary infinity
    inf_mask = np.isinf(np.real(zvp / 1j))
    d_np[inf_mask] = 0.0

    return jnp.array(d_np.reshape(np.array(zz).shape), dtype=jnp.complex128)


def _binom(n: int, k: int) -> float:
    """Binomial coefficient C(n,k) as float."""
    from math import comb
    return float(comb(n, k))


def _diff_cot_scalar(t: np.ndarray, n: int) -> np.ndarray:
    """N-th derivative of cot evaluated at array t."""
    # Use the derivative polynomials recursion (vectorized)
    x = np.tan(t + np.pi / 2)
    orig_shape = x.shape
    x_flat = x.ravel()
    sz = len(x_flat)

    P = np.zeros((sz, n + 1), dtype=complex)
    P[:, 0] = -x_flat

    for k in range(n):
        # P[k+1] = -sum_{l=0}^{k} C(k,l) P[l] * P[k-l] - delta_{k,0}
        acc = np.zeros(sz, dtype=complex)
        for l_idx in range(k + 1):
            acc += _binom(k, l_idx) * P[:, l_idx] * P[:, k - l_idx]
        P[:, k + 1] = -acc - (1.0 if k == 0 else 0.0)

    return P[:, n].reshape(orig_shape)


def _diff_csc(t: np.ndarray, n: int) -> np.ndarray:
    """N-th derivative of csc(t)."""
    return (0.5 ** n) * _diff_cot_scalar(t / 2, n) - _diff_cot_scalar(t, n)


def _diff_sin(t: np.ndarray, n: int) -> np.ndarray:
    """N-th derivative of sin(t)."""
    return np.sin(t + n * np.pi / 2)


def _diff_tan(t: np.ndarray, n: int) -> np.ndarray:
    """N-th derivative of tan(t)."""
    return -_diff_cot_scalar(t - np.pi / 2, n)


def _diff_cst(t: np.ndarray, n: int, form: str) -> np.ndarray:
    """N-th derivative of the basis function (cot or csc)."""
    if form == "even":
        return _diff_cot_scalar(t, n)
    else:
        return _diff_csc(t, n)


def _diff_cst_inv(t: np.ndarray, n: int, form: str) -> np.ndarray:
    """N-th derivative of the inverse basis function (tan or sin)."""
    if form == "even":
        return _diff_tan(t, n)
    else:
        return _diff_sin(t, n)


def trigBaryWeights(xk):
    """Barycentric weights for trigonometric interpolation at nodes xk
    in [-pi, pi] (MATLAB trigBaryWeights).

    Provenance
    ----------
    MATLAB source : trigBaryWeights.m
    Chebfun commit: 7574c77
    """
    import numpy as _np
    x = _np.asarray(xk, dtype=float).ravel()
    n = len(x)
    if n > 1 and _np.all(_np.abs(_np.diff(x) - 2 * _np.pi / n)
                         < max(_np.max(_np.abs(x)), 1.0) * 1e-14):
        w = _np.ones(n)
        w[1::2] = -1.0
        return w
    V = _np.sin(0.5 * (x[:, None] - x[None, :]))
    _np.fill_diagonal(V, 1.0)
    VV = _np.exp(_np.sum(_np.log(_np.abs(V)), axis=0))
    w = 1.0 / (_np.prod(_np.sign(V), axis=0) * VV)
    return w / _np.max(_np.abs(w))


def trigBary(x, fvals, xk=None, dom=None):
    """Trigonometric barycentric interpolation (MATLAB trigBary):
    evaluate the trig interpolant of data {xk, fvals} at x.

    Provenance
    ----------
    MATLAB source : trigBary.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    import numpy as _np
    fvals = _np.asarray(fvals, dtype=float)
    if fvals.ndim == 1:
        fvals = fvals[:, None]
        squeeze = True
    else:
        squeeze = False
    n = fvals.shape[0]
    x = _np.asarray(x, dtype=float).ravel()
    if dom is None:
        dom = (-_np.pi, _np.pi)
    a, b = float(dom[0]), float(dom[1])
    if xk is None:
        xk = a + (b - a) * _np.arange(n) / n
    xk = _np.asarray(xk, dtype=float).ravel()
    # map to [-pi, pi]
    xk = _np.pi / (b - a) * (2 * xk - a - b)
    xm = _np.pi / (b - a) * (2 * x - a - b)
    # remove periodic endpoint
    if n > 1 and abs(xk[0] + _np.pi) + abs(xk[-1] - _np.pi) \
            < 4 * _np.pi * 1e-15:
        fvals = fvals.copy()
        fvals[0, :] = 0.5 * (fvals[0, :] + fvals[-1, :])
        fvals = fvals[:-1, :]
        xk = xk[:-1]
        n -= 1
    vk = trigBaryWeights(xk)
    if n == 1:
        out = _np.repeat(fvals, len(xm), axis=0)
        return out[:, 0] if squeeze else out
    if n % 2 == 0:
        s = _np.sum(xk)
        # MATLAB rem keeps the sign of s -- np.fmod, NOT np.remainder
        c = 0.0 if abs(_np.fmod(s, _np.pi)) < 4 * _np.pi * 1e-15 \
            else 1.0 / _np.tan(s / 2.0)

        def ctsc(t):
            return 1.0 / _np.tan(t) + c
    else:
        def ctsc(t):
            return 1.0 / _np.sin(t)
    num = _np.zeros((len(xm), fvals.shape[1]))
    den = _np.zeros((len(xm), 1))
    with _np.errstate(divide="ignore", invalid="ignore"):
        for j in range(n):
            tmp = vk[j] * ctsc((xm - xk[j]) / 2.0)
            num += tmp[:, None] * fvals[j, :][None, :]
            den += tmp[:, None]
        out = num / den
    # clean up NaNs at exact nodes
    bad = _np.nonzero(~_np.isfinite(out[:, 0]))[0]
    for k in bad:
        idx = _np.nonzero(xm[k] == xk)[0]
        if len(idx):
            out[k, :] = fvals[idx[0], :]
    return out[:, 0] if squeeze else out
