"""Barycentric interpolation utilities.

Translated from MATLAB Chebfun (commit 7574c77): bary.m, trigBary.m,
baryWeights.m, trigBaryWeights.m, barymat.m.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Barycentric weights for Chebyshev points of the 2nd kind
# ---------------------------------------------------------------------------

def cheb_bary_weights(n: int) -> jnp.ndarray:
    """Barycentric weights for n Chebyshev points of the 2nd kind.

    The weights have the simple explicit form: alternating +/-1 with
    half-weight at the endpoints. They are normalised so that
    ``jnp.max(jnp.abs(w)) == 1`` and the last entry is positive.

    Parameters
    ----------
    n : int
        Number of Chebyshev points.

    Returns
    -------
    v : jnp.ndarray, shape (n,)
        Barycentric weights.

    Provenance
    ----------
    MATLAB source : @chebtech2/barywts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: Thm 5.2 of Trefethen, *Approximation Theory and
        Approximation Practice*, SIAM, 2013.

    See Also
    --------
    bary, bary_weights, barymat
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([1.0], dtype=jnp.float64)

    # General case: alternating +/-1 with half-weight at endpoints.
    # MATLAB: v = [ones(n-1,1); 0.5]; v(end-1:-2:1) = -1; v(1) = 0.5*v(1)
    #
    # Step 1: v = [1, 1, ..., 1, 0.5]  (length n)
    # Step 2: v[n-2], v[n-4], ..., v[0] = -1  (0-based; every other from n-2)
    # Step 3: v[0] *= 0.5
    #
    # The result is v[k] = (-1)^(n-1-k) for interior k, with endpoints halved.
    # Build directly using (-1)^(n-1-k):
    v = jnp.ones(n, dtype=jnp.float64)
    v = v.at[-1].set(0.5)
    # Set 0-based indices n-2, n-4, ..., 0 to -1
    # These indices have the same parity as n (0-based n-2 has parity of n).
    # An index j is in this set iff j <= n-2 and (n-2-j) % 2 == 0, i.e. j%2 == n%2
    idx = jnp.arange(n, dtype=jnp.int32)
    negate_mask = (idx % 2 == (n % 2)) & (idx <= n - 2)
    v = jnp.where(negate_mask, -1.0, v)
    # Halve the first entry
    v = v.at[0].set(v[0] * 0.5)
    return v


# ---------------------------------------------------------------------------
# General barycentric weights for arbitrary nodes
# ---------------------------------------------------------------------------

@jax.jit
def bary_weights(x: jnp.ndarray) -> jnp.ndarray:
    """Barycentric weights for arbitrary interpolation nodes.

    Computes scaled barycentric weights for the nodes in ``x`` such that
    ``jnp.max(jnp.abs(w)) == 1``.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Interpolation nodes (real).

    Returns
    -------
    w : jnp.ndarray, shape (n,)
        Barycentric weights scaled so that ``max(|w|) == 1``.

    Notes
    -----
    Uses the log-sum-exp trick to avoid overflow/underflow for large n.
    The capacity scaling ``C = 4 / (max(x) - min(x))`` improves numerical
    stability.

    Provenance
    ----------
    MATLAB source : baryWeights.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    bary, cheb_bary_weights, barymat
    """
    n = x.shape[0]

    # Capacity scaling
    C = 4.0 / (jnp.max(x) - jnp.min(x))

    # Difference matrix: V[i,j] = C*(x[i] - x[j]), with diagonal set to 1
    V = C * (x[:, None] - x[None, :])
    V = V.at[jnp.diag_indices(n)].set(1.0)

    # Log-sum of absolute differences along columns (matching MATLAB convention:
    # sum(log(abs(V))) sums along dim 1, i.e. down columns)
    log_abs_V = jnp.sum(jnp.log(jnp.abs(V)), axis=0)

    # Sign: product of signs along each column (MATLAB: prod(sign(V)))
    sign_V = jnp.prod(jnp.sign(V), axis=0)

    # Weights
    w = 1.0 / (sign_V * jnp.exp(log_abs_V))

    # Normalise
    w = w / jnp.max(jnp.abs(w))
    return w


# ---------------------------------------------------------------------------
# Trigonometric barycentric weights
# ---------------------------------------------------------------------------

@jax.jit
def trig_bary_weights(x: jnp.ndarray) -> jnp.ndarray:
    """Barycentric weights for trigonometric interpolation.

    Computes scaled barycentric weights for 2*pi-periodic trigonometric
    interpolation at the nodes ``x``. Weights are normalised so that
    ``jnp.max(jnp.abs(w)) == 1``.

    For equispaced nodes in [-pi, pi), the weights simplify to
    alternating +/-1.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Interpolation nodes in [-pi, pi].

    Returns
    -------
    w : jnp.ndarray, shape (n,)
        Trigonometric barycentric weights.

    References
    ----------
    .. [1] Berrut, "Baryzentrische Formeln zur trigonometrischen
       Interpolation (I)", ZAMP 35.1, 1984, pp. 91--105.
    .. [2] Henrici, "Barycentric formulas for interpolating trigonometric
       polynomials and their conjugates", Numer. Math. 33.2, 1979,
       pp. 225--234.

    Provenance
    ----------
    MATLAB source : trigBaryWeights.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    trig_bary, bary_weights
    """
    n = x.shape[0]

    # Difference matrix: V[i,j] = sin(0.5*(x[i] - x[j])), diagonal = 1
    V = jnp.sin(0.5 * (x[:, None] - x[None, :]))
    V = V.at[jnp.diag_indices(n)].set(1.0)

    # Log-sum of absolute differences along columns (matching MATLAB convention)
    log_abs_V = jnp.sum(jnp.log(jnp.abs(V)), axis=0)

    # Sign: product of signs along each column (matching MATLAB convention)
    sign_V = jnp.prod(jnp.sign(V), axis=0)

    # Weights
    w = 1.0 / (sign_V * jnp.exp(log_abs_V))

    # Normalise
    w = w / jnp.max(jnp.abs(w))
    return w


# ---------------------------------------------------------------------------
# Barycentric interpolation (polynomial)
# ---------------------------------------------------------------------------

@jax.jit
def bary(x: jnp.ndarray,
         fvals: jnp.ndarray,
         xk: jnp.ndarray,
         vk: jnp.ndarray) -> jnp.ndarray:
    """Barycentric interpolation formula (2nd form).

    Evaluates the polynomial interpolant of the data ``(xk, fvals)`` at the
    points ``x`` using barycentric weights ``vk``.

    This is the core evaluation routine for Chebyshev interpolants and is
    designed to be JIT-compiled and differentiable.

    Parameters
    ----------
    x : jnp.ndarray, shape (m,)
        Evaluation points.
    fvals : jnp.ndarray, shape (n,)
        Function values at the interpolation nodes ``xk``.
    xk : jnp.ndarray, shape (n,)
        Interpolation nodes.
    vk : jnp.ndarray, shape (n,)
        Barycentric weights corresponding to ``xk``.

    Returns
    -------
    fx : jnp.ndarray, shape (m,)
        Interpolated values at ``x``.

    Notes
    -----
    The 2nd barycentric formula is:

    .. math::

        p(x) = \\frac{\\sum_{k=0}^{n-1} \\frac{w_k}{x - x_k} f_k}
                     {\\sum_{k=0}^{n-1} \\frac{w_k}{x - x_k}}

    When ``x`` coincides with a node ``xk[j]``, the formula produces 0/0.
    This implementation detects such cases and returns ``fvals[j]`` directly.

    References
    ----------
    .. [1] Berrut & Trefethen, "Barycentric Lagrange Interpolation",
       SIAM Review, 46(3), 2004, pp. 501--517.

    Provenance
    ----------
    MATLAB source : bary.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    bary_weights, cheb_bary_weights, barymat, trig_bary
    """
    # diff[i, k] = x[i] - xk[k], shape (m, n)
    diff = x[:, None] - xk[None, :]

    # Quotients: vk[k] / (x[i] - xk[k]), shape (m, n)
    quot = vk[None, :] / diff

    # Numerator and denominator of the barycentric formula
    numer = jnp.dot(quot, fvals)   # shape (m,)
    denom = jnp.sum(quot, axis=1)  # shape (m,)

    fx = numer / denom

    # Fix NaNs where x == xk (0/0 case).
    # For each evaluation point, check if it coincides with any node.
    # If so, replace with the corresponding function value.
    exact_match = jnp.abs(diff) == 0.0  # shape (m, n)
    has_match = jnp.any(exact_match, axis=1)  # shape (m,)
    # Index of the matching node (argmax of boolean gives first True)
    match_idx = jnp.argmax(exact_match, axis=1)  # shape (m,)
    matched_val = fvals[match_idx]

    fx = jnp.where(has_match, matched_val, fx)
    return fx


# ---------------------------------------------------------------------------
# Trigonometric barycentric interpolation
# ---------------------------------------------------------------------------

def trig_bary(x: jnp.ndarray,
              fvals: jnp.ndarray,
              xk: jnp.ndarray,
              dom: jnp.ndarray | None = None) -> jnp.ndarray:
    """Trigonometric barycentric interpolation formula.

    Evaluates a trigonometric interpolant of the data ``(xk, fvals)`` at the
    points ``x``. The interpolant is periodic on the domain ``dom``.

    Parameters
    ----------
    x : jnp.ndarray, shape (m,)
        Evaluation points.
    fvals : jnp.ndarray, shape (n,)
        Function values at the interpolation nodes ``xk``.
    xk : jnp.ndarray, shape (n,)
        Interpolation nodes.
    dom : jnp.ndarray, shape (2,), optional
        Domain ``[a, b]``. Default is ``[-pi, pi]``.

    Returns
    -------
    fx : jnp.ndarray, shape (m,)
        Interpolated values at ``x``.

    Notes
    -----
    The trigonometric barycentric formula uses either ``csc`` or ``cot``
    depending on whether the number of nodes is odd or even, following
    Berrut [1] and Henrici [2].

    When ``dom`` is provided as a ``jnp.ndarray``, this function is
    JIT-compatible via the internal ``_trig_bary_core``. Call with an
    explicit ``dom`` argument inside JIT boundaries.

    References
    ----------
    .. [1] Berrut, "Baryzentrische Formeln zur trigonometrischen
       Interpolation (I)", ZAMP 35.1, 1984, pp. 91--105.
    .. [2] Henrici, "Barycentric formulas for interpolating trigonometric
       polynomials and their conjugates", Numer. Math. 33.2, 1979,
       pp. 225--234.

    Provenance
    ----------
    MATLAB source : trigBary.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    trig_bary_weights, bary
    """
    if dom is None:
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
    return _trig_bary_core(x, fvals, xk, dom)


@jax.jit
def _trig_bary_core(x: jnp.ndarray,
                    fvals: jnp.ndarray,
                    xk: jnp.ndarray,
                    dom: jnp.ndarray) -> jnp.ndarray:
    """Core JIT-compiled trigonometric barycentric interpolation."""
    n = xk.shape[0]
    a = dom[0]
    b = dom[1]

    # Map to [-pi, pi]
    scale = jnp.pi / (b - a)
    xk_mapped = scale * (2.0 * xk - a - b)
    x_mapped = scale * (2.0 * x - a - b)

    # Compute trigonometric barycentric weights
    vk = trig_bary_weights(xk_mapped)

    # diff[i, k] = (x_mapped[i] - xk_mapped[k]) / 2, shape (m, n)
    half_diff = (x_mapped[:, None] - xk_mapped[None, :]) / 2.0

    # For even n: use cot; for odd n: use csc
    # We use lax.cond to be JIT-compatible
    # Even n: ctsc(t) = cot(t) + c, where c accounts for node asymmetry
    # Odd n:  ctsc(t) = csc(t)
    #
    # For even n, c = cot(sum(xk)/2) if sum(xk) is not a multiple of pi, else 0
    s = jnp.sum(xk_mapped)
    c = jax.lax.cond(
        n % 2 == 0,
        lambda _: jnp.where(jnp.abs(s % jnp.pi) < 4.0 * jnp.pi * 1e-15,
                             0.0,
                             1.0 / jnp.tan(s / 2.0)),
        lambda _: 0.0,  # placeholder, not used for odd n
        operand=None,
    )

    # Compute the kernel function value for each (i, k) pair
    # Even n: vk[k] * (cot(half_diff) + c)
    # Odd n:  vk[k] * csc(half_diff) = vk[k] / sin(half_diff)
    is_even = (n % 2 == 0)
    kernel = jnp.where(
        is_even,
        1.0 / jnp.tan(half_diff) + c,   # cot + c
        1.0 / jnp.sin(half_diff),        # csc
    )

    # Weighted kernel: shape (m, n)
    weighted = vk[None, :] * kernel

    # Numerator and denominator
    numer = jnp.dot(weighted, fvals)    # shape (m,)
    denom = jnp.sum(weighted, axis=1)   # shape (m,)

    fx = numer / denom

    # Fix NaNs where x coincides with a node
    exact_match = jnp.abs(jnp.sin(half_diff)) < 1e-14  # shape (m, n)
    has_match = jnp.any(exact_match, axis=1)
    match_idx = jnp.argmax(exact_match, axis=1)
    matched_val = fvals[match_idx]

    fx = jnp.where(has_match, matched_val, fx)
    return fx


# ---------------------------------------------------------------------------
# Barycentric interpolation matrix
# ---------------------------------------------------------------------------

def barymat(y: jnp.ndarray,
            x: jnp.ndarray,
            w: jnp.ndarray | None = None) -> jnp.ndarray:
    """Barycentric interpolation matrix.

    Constructs the ``(M, N)`` matrix ``B`` such that ``B @ f`` interpolates
    the data ``(x, f)`` at the points ``y``, using the 2nd-kind barycentric
    formula with weights ``w``.

    Parameters
    ----------
    y : jnp.ndarray, shape (M,)
        Target (evaluation) grid.
    x : jnp.ndarray, shape (N,)
        Source (interpolation) grid.
    w : jnp.ndarray, shape (N,), optional
        Barycentric weights. If ``None``, uses Chebyshev 2nd-kind weights
        (alternating +/-1 with halved endpoints).

    Returns
    -------
    B : jnp.ndarray, shape (M, N)
        Interpolation matrix.

    Notes
    -----
    Where ``y[j] == x[k]`` (i.e., an evaluation point coincides with a
    source node), the matrix has a 1 in position ``(j, k)`` and 0 elsewhere
    in that row, avoiding the 0/0 singularity.

    When ``w`` is provided, this function is fully JIT-compatible via the
    internal ``_barymat_core``. Call with an explicit ``w`` argument inside
    JIT boundaries.

    Provenance
    ----------
    MATLAB source : barymat.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    bary, bary_weights, cheb_bary_weights
    """
    if w is None:
        N = x.shape[0]
        w = jnp.ones(N, dtype=jnp.float64)
        w = w.at[1::2].set(-1.0)
        w = w.at[0].set(0.5 * w[0])
        w = w.at[-1].set(0.5 * w[-1])
    return _barymat_core(y, x, w)


@jax.jit
def _barymat_core(y: jnp.ndarray,
                  x: jnp.ndarray,
                  w: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled core of barymat."""
    M = y.shape[0]
    N = x.shape[0]

    # Difference matrix: D[j, k] = y[j] - x[k], shape (M, N)
    D = y[:, None] - x[None, :]

    # B[j, k] = w[k] / D[j, k]
    B = w[None, :] / D

    # Normalise each row: B[j, :] /= sum(B[j, :])
    row_sums = jnp.sum(B, axis=1, keepdims=True)
    B = B / row_sums

    # Fix NaN rows (where y[j] == x[k], producing 0/0).
    # Replace entire NaN rows with the appropriate identity row.
    # A row is NaN if any entry in D is exactly 0.
    exact_match = (D == 0.0)  # shape (M, N)
    has_match = jnp.any(exact_match, axis=1)  # shape (M,)

    # Build the "identity" rows: one-hot at the matching column
    # For rows with a match, set B[j, :] = exact_match[j, :] (which is one-hot)
    identity_rows = exact_match.astype(jnp.float64)

    # Use where row-by-row
    B = jnp.where(has_match[:, None], identity_rows, B)

    return B
