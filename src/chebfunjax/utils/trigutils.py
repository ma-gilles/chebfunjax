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


def trigpoly(n: int | jnp.ndarray, domain=None):
    """Construct Chebfun's periodic trigonometric polynomial(s).

    The Fourier coefficient column for each degree ``n[k]`` has a one at
    row ``max(abs(n)) - n[k]`` (zero based), following the literal
    ``trigpoly.m`` coefficient indexing. Thus the default-domain function is
    ``exp(-1j*pi*n*x)``. A column-shaped degree input marks the resulting
    Chebfun transposed, matching MATLAB. The default domain follows ``ChebfunPref``.

    Empty degree input returns ``Chebfun.empty()``. Coefficients are passed
    directly to the trig constructor; no sampled-value fitting is performed.
    This factory is eager because degree and domain checks and the maximum
    degree are materialized to allocate the statically shaped coefficient
    array; evaluation of the resulting Chebfun uses JAX.

    Parameters
    ----------
    n : integer scalar or array_like
        Degree(s), arranged as a row or column vector.
    domain : sequence, optional
        Finite interval. Interior breakpoints and unbounded intervals follow
        the MATLAB source errors.

    Returns
    -------
    Chebfun
        A periodic Chebfun containing the requested degree(s).

    Provenance
    ----------
    MATLAB source : trigpoly.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and The
        Chebfun Developers.

    See Also
    --------
    diffbarytrig
    """
    try:
        n_input = jnp.asarray(n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "CHEBFUN:trigpoly:integern: The first argument must be a vector of integers."
        ) from exc
    if jnp.iscomplexobj(n_input):
        raise ValueError(
            "CHEBFUN:trigpoly:integern: The first argument must be a vector of integers."
        )
    try:
        n_float = jnp.asarray(n_input, dtype=jnp.float64)
        valid_integer_input = bool(jnp.all(
            jnp.isfinite(n_float) & (n_float == jnp.round(n_float))
        ))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "CHEBFUN:trigpoly:integern: The first argument must be a vector of integers."
        ) from exc
    if not valid_integer_input:
        raise ValueError(
            "CHEBFUN:trigpoly:integern: The first argument must be a vector of integers."
        )
    degrees = jnp.ravel(jnp.round(n_float).astype(jnp.int64), order="F")

    if domain is None:
        from chebfunjax.chebpref import ChebfunPref
        domain = ChebfunPref().domain
    try:
        d = jnp.ravel(jnp.asarray(domain, dtype=jnp.float64))
    except (TypeError, ValueError) as exc:
        raise ValueError("domain must contain finite interval endpoints") from exc
    if bool(jnp.any(jnp.isinf(d))):
        raise ValueError(
            "CHEBFUN:trigpoly:infdomain: Trigonometric polynomial construction "
            "is not allowed on unbounded domain."
        )
    if d.size > 2:
        raise ValueError(
            "CHEBFUN:trigpoly:breakpoints: Trigonometric polynomials can not be "
            "constructed on domains with break points."
        )

    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

    if degrees.size == 0:
        return Chebfun.empty()
    if d.size < 2:
        raise ValueError("domain must contain two endpoints")

    width = int(jnp.max(jnp.abs(degrees)))
    rows = width - degrees
    coeffs = jnp.zeros((2 * width + 1, degrees.size), dtype=jnp.float64)
    coeffs = coeffs.at[rows, jnp.arange(degrees.size)].set(1.0)
    result = chebfun(
        coeffs, domain=(float(d[0]), float(d[-1])), coeffs=True, trig=True
    )
    if n_input.ndim >= 2 and n_input.shape[0] > 1:
        return result.transpose()
    return result


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
    # Weight magnitudes share a common factor. Normalize their logarithms
    # before exponentiation to avoid 0 -> inf -> NaN for clustered nodes.
    x = jnp.asarray(xk, dtype=jnp.float64).ravel()
    n = len(x)
    equispaced = jnp.all(jnp.abs(jnp.diff(x) - 2*jnp.pi/n)
                     < jnp.max(jnp.abs(x)) * jnp.finfo(jnp.float64).eps)
    import jax

    def arbitrary_nodes(nodes):
        V = jnp.sin(0.5 * (nodes[:, None] - nodes[None, :]))
        V = V.at[jnp.diag_indices(n)].set(1.0)
        log_weights = -jnp.sum(jnp.log(jnp.abs(V)), axis=0)
        signs = jnp.prod(jnp.sign(V), axis=0)
        return signs * jnp.exp(log_weights - jnp.max(log_weights))

    def uniform_nodes(nodes):
        return jnp.where(jnp.arange(n) % 2, -1.0, 1.0)

    return jax.lax.cond(equispaced, uniform_nodes, arbitrary_nodes, x)


def trigBary(x, fvals, xk=None, dom=None):
    """Evaluate the native second-form trigonometric interpolant with JAX.

    Scalar data preserve the evaluation shape; two-dimensional evaluation
    arrays of multiple polynomials use native horizontal column stacking.
    Domain and nodes may be traced. Eager invalid nodes raise ValueError;
    node range validation cannot raise a Python exception for traced values.
    Default nodes span the explicit domain (existing Python extension).
    Constant/NaN data use the same final shape adapter as ordinary data;
    native early-return shape quirks are normalized for Python callers.

    Provenance
    ----------
    MATLAB source: trigBary.m, Chebfun commit 7574c77.
    Original authors: Copyright 2017 by The University of Oxford.
    """
    import jax

    x = jnp.asarray(x, dtype=jnp.float64)
    values = jnp.asarray(fvals)
    values = values.astype(jnp.result_type(values.dtype, jnp.float64))
    scalar_data = values.ndim == 1
    if scalar_data:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("trigBary requires a nonempty vector or matrix of values")
    original_n, m = values.shape
    if m > 1 and x.ndim > 2:
        raise ValueError("CHEBFUN:trigBary:evalArrayAtNDArray")
    domain = jnp.asarray((-jnp.pi, jnp.pi) if dom is None else dom)
    a, b = domain[0], domain[1]
    if xk is None:
        xk = a + (b-a)*jnp.arange(original_n)/original_n
    nodes = jnp.asarray(xk, dtype=jnp.float64)
    if nodes.ndim > 2 or (nodes.ndim == 2 and min(nodes.shape) > 1):
        raise ValueError("CHEBFUN:trigBary:xVector -- xk must be a vector")
    nodes = nodes.ravel()
    if nodes.shape[0] != original_n:
        raise ValueError("trigBary node and value lengths must agree")
    nodes = jnp.pi/(b-a)*(2*nodes-a-b)
    if not isinstance(nodes, jax.core.Tracer):
        if bool(jnp.any((nodes > jnp.pi+10*jnp.finfo(jnp.float64).eps)
                       | (nodes < -jnp.pi-10*jnp.finfo(jnp.float64).eps))):
            raise ValueError("CHEBFUN:trigbary:invalidNodes")
    if x.size == 0:
        return x
    targets = (jnp.pi/(b-a)*(2*x-a-b)).ravel(order="F")

    def evaluate(args):
        knots, data = args
        weights = trigBaryWeights(knots)
        # Native n is captured before periodic endpoint removal.
        s = jnp.sum(knots)
        c = jnp.where(jnp.abs(jnp.fmod(s,jnp.pi)) < 4*jnp.pi*jnp.finfo(jnp.float64).eps,
                      0., 1/jnp.tan(s/2))
        def step(j,state):
            num,den,matched,at_nodes = state
            delta = (targets-knots[j])/2
            member = targets == knots[j]
            kernel = (1/jnp.tan(delta)+c if original_n%2 == 0
                      else 1/jnp.sin(delta))
            ratio = weights[j]*kernel
            at_nodes = jnp.where((member & ~matched)[:,None],data[j],at_nodes)
            return (num+ratio[:,None]*data[j],den+ratio,
                    matched|member,at_nodes)
        initial = (jnp.zeros((targets.size,m),dtype=values.dtype),
                   jnp.zeros(targets.size),jnp.zeros(targets.size,dtype=bool),
                   jnp.zeros((targets.size,m),dtype=values.dtype))
        num,den,matched,at_nodes = jax.lax.fori_loop(0,knots.size,step,initial)
        out = num/den[:,None]
        cleanup = jnp.isnan(out[:,0]) & matched
        out = jnp.where(cleanup[:,None],at_nodes,out)
        # MATLAB if(any(isnan(fvals))) requires every entry of the row
        # condition to be true: every data column contains a NaN.
        nan_early_return = jnp.all(jnp.any(jnp.isnan(data),axis=0))
        return jnp.where(nan_early_return,jnp.nan,out)

    if original_n == 1:
        out = jnp.broadcast_to(values,(targets.size,m))
    else:
        periodic = jnp.linalg.norm(jnp.array([nodes[0]+jnp.pi,nodes[-1]-jnp.pi])) < 2*jnp.pi*jnp.finfo(jnp.float64).eps
        def dropped(args):
            knots,data = args
            data = data.at[0].set((data[0]+data[-1])/2)
            return evaluate((knots[:-1],data[:-1]))
        out = jax.lax.cond(periodic,dropped,evaluate,(nodes,values))
    if m == 1 and (scalar_data or x.ndim > 1):
        return out[:,0].reshape(x.shape,order="F")
    if x.ndim == 2:
        return out.reshape((x.shape[0],x.shape[1]*m),order="F")
    return out.reshape(x.shape+(m,))
