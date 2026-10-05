"""Gauss--Hermite nodes from Chebfun's Laguerre relation.

This module implements the MATLAB LAG branch. Public method selection and
probabilists' rescaling are handled by the quadrature wrapper.

Provenance
----------
MATLAB source : hermpts.m (LAG branch), lagpts.m
Chebfun commit: 7574c77
Original author: Peter Opsomer, August 2017.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from chebfunjax.utils.quadrature import lagpts


@partial(jax.jit, static_argnames=("n",))
def _hermpts_lag(n: int):
    """Return LAG physicists' Hermite nodes, weights, and barycentric weights.

    MATLAB's n=0/1 cases are handled before method dispatch, so this helper
    supports ``n >= 2``. Nodes and weights are returned as one-dimensional
    arrays; the public wrapper normalizes the quadrature weights to
    ``sqrt(pi)``. Barycentric weights have maximum absolute value one.

    Parameters
    ----------
    n : int
        Number of Hermite nodes; must be at least two.

    Returns
    -------
    x, w, v : tuple of jax.Array
        Nodes, quadrature weights, and barycentric weights.

    Provenance
    ----------
    MATLAB source : hermpts.m (LAG branch), lagpts.m
    Chebfun commit: 7574c77
    """
    if n < 2:
        raise ValueError("hermpts_lag: LAG is supported for n >= 2")

    if n % 2:
        nh = (n - 1) // 2
        x_lag, w_lag = lagpts(nh, 0.5)
        w_side = w_lag / x_lag / 2.0
        nh_float = jnp.asarray(nh, dtype=jnp.float64)
        if nh > 64:
            inv = 1.0 / nh_float
            ratio = (
                1.0 + inv / 8.0 + inv**2 / 128.0
                - 5.0 * inv**3 / 1024.0
                - 21.0 * inv**4 / 32768.0
                + 399.0 * inv**5 / 262144.0
                + 869.0 * inv**6 / 4194304.0
                - 39325.0 * inv**7 / 33554432.0
                - 334477.0 * inv**8 / 2147483648.0
                + 28717403.0 * inv**9 / 17179869184.0
                + 59697183.0 * inv**10 / 274877906944.0
            )
            w0 = jnp.pi * ratio / n * jnp.sqrt(nh_float)
        else:
            factorial_nh = jnp.prod(
                jnp.arange(1, nh + 1, dtype=jnp.float64))
            w0 = jnp.pi * factorial_nh / n / jsp.gamma(nh_float + 0.5)
        x_side = jnp.sqrt(x_lag)
        x = jnp.concatenate((-x_side[::-1], jnp.zeros((1,)), x_side))
        w = jnp.concatenate((w_side[::-1], jnp.reshape(w0, (1,)), w_side))
    else:
        nh = n // 2
        x_lag, w_lag = lagpts(nh, -0.5)
        x_side = jnp.sqrt(x_lag)
        x = jnp.concatenate((-x_side[::-1], x_side))
        w = jnp.concatenate((w_lag[::-1], w_lag)) / 2.0

    signs = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    v = signs * jnp.sqrt(w) / jnp.sqrt(jnp.max(w))
    # The public wrapper applies source top-level normalization after v is
    # formed. Unlike REC/ASY, MATLAB's LAG branch does not normalize twice.
    return x, w, v
