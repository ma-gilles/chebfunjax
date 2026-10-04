"""Legendre Gauss nodes from Chebfun's recurrence/Newton (REC) method.

This is the low-order companion to the Golub--Welsch and asymptotic
Legendre quadrature rules.  The polynomial values and derivatives are
advanced together by the three-term Legendre recurrence; Newton iteration
starts from Tricomi's positive-root approximation.

Provenance
----------
MATLAB source : legpts.m (local function ``rec``)
Chebfun commit: 7574c77
Original author: Nick Hale, July 2011.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def _legendre_and_derivative(x: jax.Array, n: int):
    """Evaluate P_n and P'_n at ``x`` with a joint recurrence."""
    p0 = jnp.ones_like(x)
    p1 = x
    dp0 = jnp.zeros_like(x)
    dp1 = jnp.ones_like(x)

    def step(k, state):
        pm2, pm1, dpm2, dpm1 = state
        kf = jnp.asarray(k, dtype=x.dtype)
        p = ((2.0 * kf + 1.0) * pm1 * x - kf * pm2) / (kf + 1.0)
        # P_{k+1}' = ((2k+1)(P_k + x P_k') - k P_{k-1}')/(k+1).
        dp = ((2.0 * kf + 1.0) * (pm1 + x * dpm1) - kf * dpm2) / (kf + 1.0)
        return pm1, p, dpm1, dp

    _, pn, _, dpn = lax.fori_loop(1, n, step, (p0, p1, dp0, dp1))
    return pn, dpn


@partial(jax.jit, static_argnames=("n",))
def _legpts_rec(n: int):
    """Return ``(x, w, v)`` for the n-point Gauss--Legendre rule on [-1, 1].

    ``n`` is static because it determines the output shapes and recurrence
    length.  The implementation uses pure JAX operations and ``lax`` loops.
    ``v`` is normalized to max(abs(v)) == 1 and alternates signs starting
    positive, matching the third output of MATLAB ``legpts``.
    """
    if n < 0:
        raise ValueError("legpts_rec: n must be nonnegative")
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return empty, empty, empty
    if n == 1:
        return (jnp.zeros((1,), dtype=jnp.float64),
                jnp.full((1,), 2.0, dtype=jnp.float64),
                jnp.ones((1,), dtype=jnp.float64))
    if n == 2:
        x = jnp.asarray([-1.0, 1.0], dtype=jnp.float64) / jnp.sqrt(3.0)
        w = jnp.ones((2,), dtype=jnp.float64)
        v = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
        return x, w, v

    dtype = jnp.float64
    odd = n % 2
    m = (n + odd) // 2
    k = jnp.arange(m, 0, -1, dtype=dtype)
    theta = jnp.pi * (4.0 * k - 1.0) / (4.0 * n + 2.0)
    x0 = (1.0 - (n - 1.0) / (8.0 * n**3)
          - (39.0 - 28.0 / jnp.sin(theta) ** 2) / (384.0 * n**4)) * jnp.cos(theta)

    dx0 = jnp.full_like(x0, jnp.inf)

    def continue_newton(state):
        count, _x, dx = state
        return (count < 10) & (jnp.max(jnp.abs(dx)) > jnp.finfo(dtype).eps)

    def newton_step(state):
        count, x, _dx = state
        p, dp = _legendre_and_derivative(x, n)
        dx = -p / dp
        return count + 1, x + dx, dx

    _, x_pos, _ = lax.while_loop(
        continue_newton, newton_step,
        (jnp.asarray(0, dtype=jnp.int32), x0, dx0),
    )
    if odd:
        x_pos = x_pos.at[0].set(0.0)

    _, dp_pos = _legendre_and_derivative(x_pos, n)
    if odd:
        reflected = -x_pos[1:][::-1]
        x = jnp.concatenate((reflected, x_pos))
        dp = jnp.concatenate((dp_pos[1:][::-1], dp_pos))
    else:
        x = jnp.concatenate((-x_pos[::-1], x_pos))
        dp = jnp.concatenate((dp_pos[::-1], dp_pos))

    w = 2.0 / ((1.0 - x**2) * dp**2)
    v = jnp.abs(1.0 / dp)
    v = v / jnp.max(v)
    signs = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    return x, w, v * signs
