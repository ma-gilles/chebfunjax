"""Quadrature points and weights.

Translated from MATLAB Chebfun (commit 7574c77): chebpts.m and related functions.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import jax.numpy as jnp


def chebpts(n: int, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points of the first or second kind on [-1, 1].

    CHEBPTS(N) returns N Chebyshev points of the 2nd kind in [-1, 1].
    CHEBPTS(N, 1) returns N Chebyshev points of the 1st kind.

    Parameters
    ----------
    n : int
        Number of points.
    kind : {1, 2}
        1 for roots of T_n (Gauss-Chebyshev), 2 for extrema of T_{n-1}
        (Clenshaw-Curtis / Chebyshev-Lobatto). Default is 2.

    Returns
    -------
    x : jnp.ndarray, shape (n,)
        Chebyshev points, ordered from -1 to 1.

    Provenance
    ----------
    MATLAB source : chebpts.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
            quadrature rules", BIT Numerical Mathematics, 46, 2006.

    See Also
    --------
    chebweights, chebpts_ab
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([0.0], dtype=jnp.float64)

    if kind == 1:
        # Roots of T_n: cos((2k-1)*pi / (2n)), k = 1..n
        k = jnp.arange(n, 0, -1, dtype=jnp.float64)
        x = jnp.cos((2 * k - 1) * jnp.pi / (2 * n))
    elif kind == 2:
        # Extrema of T_{n-1}: cos(k*pi / (n-1)), k = 0..n-1
        k = jnp.arange(n - 1, -1, -1, dtype=jnp.float64)
        x = jnp.cos(k * jnp.pi / (n - 1))
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")

    return x


def chebpts_ab(n: int, a: float, b: float, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points on the interval [a, b]."""
    x = chebpts(n, kind)
    return 0.5 * ((b - a) * x + (b + a))


def chebweights(n: int, kind: int = 2) -> jnp.ndarray:
    """Clenshaw-Curtis (kind=2) or Gauss-Chebyshev (kind=1) quadrature weights.

    Parameters
    ----------
    n : int
        Number of points.
    kind : {1, 2}
        1 for Gauss-Chebyshev weights, 2 for Clenshaw-Curtis weights.

    Returns
    -------
    w : jnp.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : chebpts.m (weights computed alongside points)
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm:
        [1] Waldvogel, "Fast construction of the Fejér and Clenshaw-Curtis
            quadrature rules", BIT Numerical Mathematics, 46, 2006.

    See Also
    --------
    chebpts
    """
    if n == 0:
        return jnp.array([], dtype=jnp.float64)
    if n == 1:
        return jnp.array([2.0], dtype=jnp.float64)

    if kind == 1:
        # Gauss-Chebyshev: w_k = pi/n for all k
        return jnp.full(n, jnp.pi / n, dtype=jnp.float64)
    elif kind == 2:
        # Clenshaw-Curtis weights via FFT (Waldvogel's algorithm)
        return _clenshaw_curtis_weights(n)
    else:
        raise ValueError(f"kind must be 1 or 2, got {kind}")


def _clenshaw_curtis_weights(n: int) -> jnp.ndarray:
    """Clenshaw-Curtis quadrature weights for n second-kind Chebyshev points.

    Uses Waldvogel's FFT-based algorithm (BIT 2006).
    Points: x_k = cos(k*pi/(n-1)), k = 0..n-1 (descending order).
    Weights satisfy: sum(w) = 2, and integrate polynomials of degree <= n-1 exactly.
    Returns weights in ascending x order (matching chebpts output).
    """
    if n == 2:
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    N = n - 1

    # Chebyshev moments: integral of T_k(x) over [-1,1]
    # = 2/(1-k^2) for even k, 0 for odd k
    # Build the first N+1 moments
    c = jnp.zeros(N + 1, dtype=jnp.float64)
    k_even = jnp.arange(0, N + 1, 2, dtype=jnp.float64)
    c = c.at[0::2].set(2.0 / (1.0 - k_even**2))

    # Mirror to get a vector of length 2N for IFFT
    # v = [c[0], c[1], ..., c[N], c[N-1], ..., c[1]]
    v = jnp.concatenate([c, c[N - 1:0:-1]])

    # IFFT gives weights / N. We want the true CC weights which sum to 2.
    # The IFFT divides by 2N (length of v), but the DCT-I normalization
    # needs division by N only, so multiply by 2.
    w = 2.0 * jnp.real(jnp.fft.ifft(v))

    # Extract the first N+1 values (theta = 0..pi)
    w = w[:N + 1]

    # Halve the endpoints (trapezoidal rule correction)
    w = w.at[0].set(w[0] / 2.0)
    w = w.at[N].set(w[N] / 2.0)

    # Reverse: MATLAB chebpts returns ascending order (-1 to 1)
    return w[::-1]
