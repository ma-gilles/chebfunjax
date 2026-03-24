# uses-numpy: Lebesgue function evaluation uses numpy loops over points
"""Lebesgue function and Lebesgue constant for polynomial interpolation.

The Lebesgue function ``λ(x)`` for a set of interpolation nodes ``x_0, …, x_{n-1}``
is defined by

    λ(x) = Σ_k |ℓ_k(x)|

where ``ℓ_k`` are the Lagrange basis polynomials.  The Lebesgue *constant* is

    Λ = max_{x ∈ [a, b]} λ(x).

The Lebesgue constant measures how much worse polynomial interpolation in
the chosen nodes can be compared to the best polynomial approximation.  For
Chebyshev nodes, ``Λ = O(log n)``; for equispaced nodes, ``Λ = O(2^n / n)``.

The barycentric formula is used for evaluation, following the algorithm in:
    Higham, "The numerical stability of barycentric Lagrange interpolation",
    IMA J. Numer. Anal., 24(4):547–556, 2004.

Translated from MATLAB Chebfun ``lebesgue.m`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

Provenance
----------
MATLAB source : lebesgue.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "lebesgue_function",
    "lebesgue_constant",
    "bary_weights",
]


# ---------------------------------------------------------------------------
# Barycentric weights
# ---------------------------------------------------------------------------


def bary_weights(x: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for an arbitrary node set.

    Uses the standard formula

        w_k = 1 / Π_{j ≠ k} (x_k - x_j)

    with the absolute value of the products to avoid catastrophic
    cancellation (signs are not needed for the Lebesgue function since we
    only need ``|ℓ_k(t)|``).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Interpolation nodes (distinct, real).

    Returns
    -------
    w : np.ndarray, shape (n,)
        Barycentric weights (un-normalised; signs are preserved).

    Examples
    --------
    >>> import numpy as np
    >>> w = bary_weights(np.array([-1.0, 0.0, 1.0]))
    >>> w.shape
    (3,)

    Provenance
    ----------
    MATLAB source : baryWeights.m
    Chebfun commit: 7574c77
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    w = np.ones(n, dtype=np.float64)
    for k in range(n):
        diffs = x[k] - np.delete(x, k)
        # Use log-sum to avoid overflow/underflow for large n
        signs = np.sign(diffs)
        log_abs = np.log(np.abs(diffs))
        sign_prod = np.prod(signs)
        log_prod = np.sum(log_abs)
        if np.isfinite(log_prod):
            w[k] = sign_prod * np.exp(-log_prod)
        else:
            # Fallback: direct product (may overflow for n > ~170)
            w[k] = sign_prod * np.exp(-log_prod) if log_prod < 0 else 0.0
    return w


# ---------------------------------------------------------------------------
# Lebesgue function
# ---------------------------------------------------------------------------


def _lebesgue_fun_at(t: float, x: np.ndarray, w: np.ndarray) -> float:
    """Evaluate the Lebesgue function at a single point *t*.

    Parameters
    ----------
    t : float
        Evaluation point.
    x : np.ndarray
        Interpolation nodes.
    w : np.ndarray
        Barycentric weights for *x*.

    Returns
    -------
    float
        Value of the Lebesgue function at *t*.
    """
    # If t coincides with a node, ℓ_k(t) = δ_{ik} => λ(t) = 1
    if np.any(x == t):
        return 1.0

    diff = w / (t - x)
    return float(np.sum(np.abs(diff)) / abs(np.sum(diff)))


def lebesgue_function(
    nodes: np.ndarray | jnp.ndarray,
    domain: tuple[float, float] = (-1.0, 1.0),
    *,
    n_eval: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lebesgue function for a set of interpolation nodes.

    Returns arrays ``(t, lam)`` where ``t`` is an equispaced evaluation grid
    over *domain* and ``lam[i]`` is the value of the Lebesgue function at
    ``t[i]``.

    Parameters
    ----------
    nodes : array-like, shape (n,)
        Interpolation nodes inside *domain*.
    domain : (float, float), default (-1, 1)
        Interval ``[a, b]`` for the Lebesgue function.
    n_eval : int, default 2001
        Number of evaluation points (odd for symmetry).

    Returns
    -------
    t : np.ndarray, shape (n_eval,)
        Equispaced evaluation grid.
    lam : np.ndarray, shape (n_eval,)
        Lebesgue function values.

    Examples
    --------
    >>> import numpy as np
    >>> from chebfunjax.utils.quadrature import chebpts
    >>> from chebfunjax.utils.lebesgue import lebesgue_function, lebesgue_constant
    >>> x = np.array(chebpts(8))
    >>> t, lam = lebesgue_function(x)
    >>> lam.min() >= 1.0  # Lebesgue function >= 1 everywhere
    True

    Provenance
    ----------
    MATLAB source : lebesgue.m (polyLebesgue / polyLebesgueFun)
    Chebfun commit: 7574c77
    """
    nodes = np.asarray(nodes, dtype=np.float64).ravel()
    a, b = float(domain[0]), float(domain[1])

    w = bary_weights(nodes)

    t = np.linspace(a, b, n_eval)
    lam = np.array([_lebesgue_fun_at(ti, nodes, w) for ti in t])

    return t, lam


def lebesgue_constant(
    nodes: np.ndarray | jnp.ndarray,
    domain: tuple[float, float] = (-1.0, 1.0),
    *,
    n_eval: int = 2001,
) -> float:
    """Compute the Lebesgue constant Λ = max_x λ(x) for a node set.

    Parameters
    ----------
    nodes : array-like, shape (n,)
        Interpolation nodes.
    domain : (float, float), default (-1, 1)
        Domain for the maximum.
    n_eval : int, default 2001
        Number of points used to estimate the maximum.

    Returns
    -------
    Lambda : float
        Lebesgue constant (approximate; limited by ``n_eval``).

    Examples
    --------
    >>> import numpy as np
    >>> from chebfunjax.utils.quadrature import chebpts
    >>> from chebfunjax.utils.lebesgue import lebesgue_constant
    >>> # Chebyshev nodes give a nearly-optimal Lebesgue constant ~ log(n)
    >>> x = np.array(chebpts(8))
    >>> lc = lebesgue_constant(x)
    >>> 1.0 < lc < 3.0  # O(log 8) ≈ 2.08
    True

    Provenance
    ----------
    MATLAB source : lebesgue.m (norm(L, inf) step)
    Chebfun commit: 7574c77
    """
    _, lam = lebesgue_function(nodes, domain=domain, n_eval=n_eval)
    return float(np.max(lam))
