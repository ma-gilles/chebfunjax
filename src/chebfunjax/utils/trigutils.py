"""Trigonometric polynomial utilities.

Translated from MATLAB Chebfun (commit 7574c77): trigpoly.m, diffbarytrig.m.
Original: Copyright 2017-2018 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
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
                binom * (0.5 ** (q - p)) * dcst_vals * fj_minus_rp * wj_np[None, :]
            )

        rp[:, p] = np.sum(DR, axis=(1, 2)) / rpDen

        # Differentiation matrix D^{(p)}
        D_new = np.zeros((m, m), dtype=complex)
        for q in range(1, p + 1):
            binom = _binom(p, q)
            # diff_cst at support point differences
            diff_zj = (zj_np[:, None] - zj_np[None, :]) / 2.0  # (m, m)
            dcst_inv_diag = _diff_cst_inv(diff_zj, q, form)  # (m, m)
            Dp_q = D_list[p - q]  # (m, m)
            # First sum: diagonal terms
            first_sum = (wj_np[:, None] / wj_np[None, :]) * np.sum(
                np.eye(m)[:, :, None] * Dp_q[:, :, None] * (0.5 ** (-q)) * dcst_inv_diag[:, :, None],
                axis=1,
            )
            # Second sum: off-diagonal terms
            second_sum = Dp_q * (0.5 ** (-q)) * dcst_inv_diag

            contrib = cst(diff_zj) * binom * (first_sum - second_sum)
            D_new += contrib

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


def _diff_cot(t: np.ndarray, n: int) -> np.ndarray:
    """N-th derivative of cot(t) using derivative polynomials."""
    x = np.tan(t + np.pi / 2)
    shape = list(x.shape) + [n + 1]
    P = np.zeros(shape, dtype=complex)
    P[..., 0] = -x
    for k in range(n):
        ell = np.arange(k + 1).reshape([1] * len(x.shape) + [-1])
        Pk = P[..., :k + 1]
        Pkl = Pk[..., :k + 1]
        Pkm = Pk[..., ::-1][..., :k + 1]
        # Derivative polynomial recurrence
        from math import factorial, comb
        fact_k = factorial(k)
        for l_idx in range(k + 1):
            # sum term
            pass
        # Simplified: use recursion
        l_vec = np.arange(k + 1)
        coeffs = fact_k / (np.array([factorial(l) * factorial(k - l) for l in l_vec]))
        # P[k+1] = -sum_{l=0}^k C(k,l) P[l] * P[k-l] - delta(k,0)
        P[..., k + 1] = -np.einsum('...i,...i,i->', P[..., :k + 1], P[..., k::-1],
                                   np.array([_binom(k, l) for l in range(k + 1)])) \
            * np.ones(x.shape) - (1.0 if k == 0 else 0.0)
    return P[..., n]


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
