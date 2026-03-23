"""Quadrature points and weights.

Translated from MATLAB Chebfun: chebpts.m, legpts.m, and related functions.
Reference: /scratch/gpfs/GILLES/mg6942/chebfun_matlab_ref/chebpts.m
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def chebpts(n: int, kind: int = 2) -> jnp.ndarray:
    """Chebyshev points of the first or second kind on [-1, 1].

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
    """Clenshaw-Curtis quadrature weights for n points (second kind).

    Uses the FFT-based algorithm. Reference: Waldvogel (2006).
    """
    if n == 2:
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    N = n - 1
    # theta_k = k*pi/N for k = 0..N
    c = jnp.zeros(n, dtype=jnp.float64)
    # Build the weight-generating function in Chebyshev coefficient space
    # w_k = (2/N) * sum_{j=0}^{N/2} b_j * cos(2*j*k*pi/N)
    # where b_j = 2/(1 - 4j^2) for j >= 1, b_0 = 1
    m = N // 2
    j = jnp.arange(1, m + 1, dtype=jnp.float64)
    bj = 2.0 / (1.0 - 4.0 * j**2)

    # Use the DCT relationship
    # Place coefficients for the cosine series
    c = c.at[0].set(1.0)
    c = c.at[1:m + 1].set(bj)
    if N % 2 == 0:
        c = c.at[m].set(bj[-1] / 2)  # Nyquist correction

    # Mirror for DCT-I
    if N % 2 == 0:
        v = jnp.concatenate([c[:m + 1], c[m - 1:0:-1]])
    else:
        v = jnp.concatenate([c[:m + 1], c[m:0:-1]])

    w = jnp.real(jnp.fft.ifft(v))
    weights = jnp.zeros(n, dtype=jnp.float64)
    weights = weights.at[0].set(w[0] / 2)
    weights = weights.at[1:n - 1].set(w[1:n - 1])
    weights = weights.at[n - 1].set(w[0] / 2)  # symmetry: w[N] = w[0]

    # Normalize
    weights = weights * (2.0 / N)

    # Reverse to match point ordering (ascending)
    return weights[::-1]
