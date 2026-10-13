"""Trigonometric polynomial utilities.

Translated from MATLAB Chebfun (commit 7574c77): trigpoly.m, diffbarytrig.m.
Original: Copyright 2017-2018 by The University of Oxford and The Chebfun
Developers.  See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import math
from numbers import Real

import jax.numpy as jnp

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

def diffbarytrig(
    zz: jnp.ndarray,
    zj: jnp.ndarray,
    fj: jnp.ndarray,
    wj: jnp.ndarray,
    N: int | str = 1,
    form: str | int = "odd",
) -> jnp.ndarray:
    """Derivative of a trigonometric rational function in barycentric form.

    DIFFBARYTRIG(ZZ, ZJ, FJ, WJ) returns the first derivative of an odd
    trigonometric barycentric rational function. ``N`` selects the derivative
    order and ``form`` selects the odd (csc) or even (cot) basis. The two
    optional arguments may be supplied in either order. Form matching is
    case-insensitive and accepts the native ``even``/``odd`` prefixes.

    Parameters
    ----------
    zz : array_like
        Evaluation points; output preserves this shape, including complex
        points and imaginary infinity.
    zj, fj, wj : array_like
        Support points, sampled values, and barycentric weights.
    N : int, default 1
        Static nonnegative derivative order. Zero raises the native
        ``Use revaltrig.m for function evaluation`` message.
    form : str, default ``"odd"``
        ``"odd"``/``"even"`` kernel selection, case-insensitive and prefix
        matched as in MATLAB.

    Returns
    -------
    jax.Array
        Complex derivative values with the same shape as ``zz``.

    Notes
    -----
    Evaluation and support-point repair use the source formulas directly.
    Python scalar derivative order and form must be static under JIT. Unknown
    option strings are rejected instead of reproducing the native parser's
    no-progress loop. The native parser's bare ``isinteger`` expression also
    prints a value to MATLAB's command window; this array API intentionally
    has no corresponding stdout side effect.

    Provenance
    ----------
    MATLAB source : diffbarytrig.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2018 by The University of Oxford
        and The Chebfun Developers.
    """
    # MATLAB accepts optional N and form in either order.
    if isinstance(N, str):
        if isinstance(form, str):
            if form != "odd":
                raise ValueError("diffbarytrig accepts at most one form and one derivative order")
            form, N = N, 1
        else:
            form, N = N, form
    if not isinstance(form, str):
        raise ValueError("form must begin with 'odd' or 'even'")
    form_key = form.casefold()
    if form_key.startswith("even"):
        normalized_form = "even"
    elif form_key.startswith("odd"):
        normalized_form = "odd"
    else:
        raise ValueError("form must begin with 'odd' or 'even'")
    try:
        if N == 0:
            raise ValueError("Use revaltrig.m for function evaluation")
        if isinstance(N, bool) or not isinstance(N, Real):
            raise TypeError
        n_value = float(N)
    except (TypeError, ValueError, OverflowError):
        if N == 0:
            raise ValueError("Use revaltrig.m for function evaluation") from None
        raise ValueError("derivative order N must be a static integer") from None
    if not math.isfinite(n_value) or n_value != round(n_value):
        raise ValueError("derivative order N must be a static integer")
    N = int(n_value)
    if N < 0:
        raise ValueError("derivative order N must be positive")

    zz_arr = jnp.asarray(zz, dtype=jnp.complex128)
    original_shape = zz_arr.shape
    zj_arr = jnp.asarray(zj, dtype=jnp.complex128).reshape((-1,))
    fj_arr = jnp.asarray(fj, dtype=jnp.complex128).reshape((-1,))
    wj_arr = jnp.asarray(wj, dtype=jnp.complex128).reshape((-1,))
    zv = zz_arr.reshape((-1,))
    m = zj_arr.size

    # Projection onto the first period window, as in source lines 40-41.
    zvp = zv - 2.0 * jnp.pi * jnp.floor(jnp.real(zv / (2.0 * jnp.pi)))

    diff_half = (zvp[:, None] - zj_arr[None, :]) / 2.0
    CC = _cst(diff_half, normalized_form)
    rp_den = CC @ wj_arr
    rn = (CC @ (wj_arr * fj_arr)) / rp_den
    rp_values = [rn]
    d_matrices = [jnp.eye(m, dtype=jnp.complex128)]
    diff_zj = (zj_arr[:, None] - zj_arr[None, :]) / 2.0
    cst_diff_zj = _cst(diff_zj, normalized_form)

    for p in range(1, N + 1):
        dr_terms = []
        for q in range(p):
            binom = _binom(p, q)
            dcst_vals = _diff_cst(diff_half, p - q, normalized_form)
            fj_minus_rp = (fj_arr[None, :] if q == 0 else 0.0) - rp_values[q][:, None]
            dr_terms.append(
                binom * (2.0 ** (q - p)) * dcst_vals * fj_minus_rp * wj_arr[None, :]
            )
        # Source sums first across supports and then across derivative terms.
        dr_stack = jnp.stack(dr_terms, axis=-1)
        rp_p = jnp.sum(jnp.sum(dr_stack, axis=1), axis=1) / rp_den
        rp_values.append(rp_p)

        matrix_terms = []
        for q in range(1, p + 1):
            binom = _binom(p, q)
            d_pq = d_matrices[p - q]
            # Source firstSum uses wj.'./wj times the row diagonal of D^(p-q).
            inv_at_zero = _diff_cst_inv(jnp.zeros((), dtype=jnp.float64), q, normalized_form)
            source_diagonal_sum = jnp.sum(
                jnp.eye(m, dtype=jnp.complex128) * d_pq, axis=1
            )
            first_sum = (
                (wj_arr[None, :] / wj_arr[:, None])
                * source_diagonal_sum[:, None]
                * (2.0 ** (-q))
                * inv_at_zero
            )
            second_sum = d_pq * (2.0 ** (-q)) * _diff_cst_inv(
                diff_zj, q, normalized_form
            )
            matrix_terms.append(binom * (first_sum - second_sum))

        # Keep MATLAB's cst(diff_zj) * sum_q(...) assembly order. Only the
        # resulting diagonal is replaced; off-diagonal singularities persist.
        d_new = cst_diff_zj * jnp.sum(jnp.stack(matrix_terms, axis=-1), axis=-1)
        d_new = d_new.at[jnp.diag_indices(m)].set(0.0)
        d_new = d_new - jnp.diag(jnp.sum(d_new, axis=1))
        d_matrices.append(d_new)

    derivative = rp_values[N]
    dzj = d_matrices[N] @ fj_arr
    support_match = zvp[:, None] == zj_arr[None, :]
    has_match = jnp.any(support_match, axis=1)
    first_match = jnp.argmax(support_match, axis=1)
    at_support = dzj[first_match]
    repair = jnp.isnan(derivative) & ~jnp.isnan(zvp) & has_match
    derivative = jnp.where(repair, at_support, derivative)

    # Native source sets the derivative to zero at imaginary infinity.
    imaginary_infinity = jnp.isinf(jnp.real(zvp / 1j))
    derivative = jnp.where(imaginary_infinity, 0.0, derivative)
    return derivative.reshape(original_shape)


def _binom(n: int, k: int) -> float:
    """Binomial coefficient C(n,k) as a Python scalar."""
    from math import comb
    return float(comb(n, k))


def _diff_cot_scalar(t: jnp.ndarray, n: int) -> jnp.ndarray:
    """N-th cotangent derivative from the source derivative polynomials."""
    x = jnp.tan(jnp.asarray(t) + jnp.pi / 2)
    shape = x.shape
    flat = x.reshape((-1,))
    p_values = jnp.zeros((flat.size, n + 1), dtype=jnp.complex128)
    p_values = p_values.at[:, 0].set(-flat)

    for k in range(n):
        acc = jnp.zeros((flat.size,), dtype=jnp.complex128)
        for ell in range(k + 1):
            acc = acc + _binom(k, ell) * p_values[:, ell] * p_values[:, k - ell]
        p_values = p_values.at[:, k + 1].set(-acc - (1.0 if k == 0 else 0.0))
    return p_values[:, n].reshape(shape)


def _diff_csc(t: jnp.ndarray, n: int) -> jnp.ndarray:
    """N-th cosecant derivative, matching native ``diffCsc``."""
    return (0.5 ** n) * _diff_cot_scalar(t / 2, n) - _diff_cot_scalar(t, n)


def _diff_sin(t: jnp.ndarray, n: int) -> jnp.ndarray:
    """N-th sine derivative."""
    return jnp.sin(t + n * jnp.pi / 2)


def _diff_tan(t: jnp.ndarray, n: int) -> jnp.ndarray:
    """N-th tangent derivative, matching native ``diffTan``."""
    return -_diff_cot_scalar(t - jnp.pi / 2, n)


def _diff_cst(t: jnp.ndarray, n: int, form: str) -> jnp.ndarray:
    """N-th derivative of the cotangent or cosecant kernel."""
    return _diff_cot_scalar(t, n) if form == "even" else _diff_csc(t, n)


def _diff_cst_inv(t: jnp.ndarray, n: int, form: str) -> jnp.ndarray:
    """N-th derivative of the tangent or sine inverse kernel."""
    return _diff_tan(t, n) if form == "even" else _diff_sin(t, n)


def _cst(t: jnp.ndarray, form: str) -> jnp.ndarray:
    """Native cotangent (even) or cosecant (odd) kernel."""
    if form == "even":
        return 1.0 / jnp.tan(t)
    return 1.0 / jnp.sin(t)


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
