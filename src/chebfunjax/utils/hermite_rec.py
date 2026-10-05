"""Recurrence/Newton Gauss--Hermite nodes from Chebfun's ``hermpts``.

This module implements only the MATLAB REC branch. Method selection,
probabilists' rescaling, and integration into the public quadrature API are
handled by the caller.

Provenance
----------
MATLAB source : hermpts.m (``HermiteInitialGuesses``, ``hermpts_rec``,
    ``hermpoly_rec``)
Chebfun commit: 7574c77
Original author: Nick Hale, July 2011.
"""

from __future__ import annotations

from functools import partial
from math import ceil, floor, ulp

import jax
import jax.numpy as jnp
from jax import lax

_AIRY_ROOTS = (
    -2.338107410459762,
    -4.087949444130970,
    -5.520559828095555,
    -6.786708090071765,
    -7.944133587120863,
    -9.022650853340979,
    -10.040174341558084,
    -11.008524303733260,
    -11.936015563236262,
    -12.828776752865757,
)


def _initial_guesses(n: int) -> jax.Array:
    """Positive-half guesses from MATLAB's Airy/Tricomi blend."""
    odd = n % 2
    m = (n - 1) // 2 if odd else n // 2
    a = 0.5 if odd else -0.5
    nu = 4.0 * m + 2.0 * a + 2.0
    j = jnp.arange(1, m + 1, dtype=jnp.float64)

    t = 3.0 * jnp.pi / 8.0 * (4.0 * j - 1.0)
    t2 = t ** (-2)
    airy_asym = -(t ** (2.0 / 3.0)) * (
        1.0 + (5.0 / 48.0) * t2 - (5.0 / 36.0) * t2**2
        + (77125.0 / 82944.0) * t2**3
        - (108056875.0 / 6967296.0) * t2**4
        + (162375596875.0 / 334430208.0) * t2**5
    )
    n_exact = min(m, len(_AIRY_ROOTS))
    airy_roots = airy_asym.at[:n_exact].set(jnp.asarray(_AIRY_ROOTS[:n_exact]))

    airy = jnp.sqrt(
        nu
        + 2.0 ** (2.0 / 3.0) * airy_roots * nu ** (1.0 / 3.0)
        + (1.0 / 5.0) * 2.0 ** (4.0 / 3.0) * airy_roots**2 * nu ** (-1.0 / 3.0)
        + (11.0 / 35.0 - a**2 - (12.0 / 175.0) * airy_roots**3) / nu
        + (16.0 / 1575.0 * airy_roots + 92.0 / 7875.0 * airy_roots**4)
        * 2.0 ** (2.0 / 3.0) * nu ** (-5.0 / 3.0)
        - (15152.0 / 3031875.0 * airy_roots**5
           + 1088.0 / 121275.0 * airy_roots**2)
        * 2.0 ** (1.0 / 3.0) * nu ** (-7.0 / 3.0)
    )[::-1]

    rhs = (4.0 * m - 4.0 * j + 3.0) / nu * jnp.pi
    t0 = jnp.full((m,), jnp.pi / 2.0, dtype=jnp.float64)

    def tricomi_step(_, t_value):
        residual = t_value - jnp.sin(t_value) - rhs
        derivative = 1.0 - jnp.cos(t_value)
        return t_value - residual / derivative

    t0 = lax.fori_loop(0, 7, tricomi_step, t0)
    tval = jnp.cos(t0 / 2.0) ** 2
    sin_guess = jnp.sqrt(
        nu * tval
        - (5.0 / (4.0 * (1.0 - tval) ** 2) - 1.0 / (1.0 - tval)
           - 1.0 + 3.0 * a**2) / (3.0 * nu)
    )

    # MATLAB uses p = .4985 + eps and one-based ceil/floor indices.
    p = 0.4985 + ulp(1.0)
    cut = floor(p * n)
    airy_start = ceil(p * n) - 1
    positive = jnp.concatenate((sin_guess[:cut], airy[airy_start:]))
    if odd:
        return jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64),
                                positive))[:m + 1]
    return positive[:m]


def _hermite_scaled_value_derivative(n: int, x: jax.Array):
    """Evaluate MATLAB's Gaussian-scaled normalized H_n and derivative."""
    hold0 = jnp.exp(-(x**2) / 4.0)
    h0 = x * hold0

    def recurrence(k, state):
        hold, h = state
        kf = jnp.asarray(k, dtype=jnp.float64)
        hnew = x * h / jnp.sqrt(kf + 1.0) - hold / jnp.sqrt(1.0 + 1.0 / kf)
        return h, hnew

    hold, hn = lax.fori_loop(1, n, recurrence, (hold0, h0))
    derivative = -x * hn + jnp.sqrt(jnp.asarray(n, dtype=jnp.float64)) * hold
    return hn, derivative


@partial(jax.jit, static_argnames=("n",))
def _hermpts_rec(n: int):
    """Return REC ``(x, w, v)`` on the physicists' Hermite convention.

    Nodes and weights are normalized so ``sum(w) == sqrt(pi)``. Barycentric
    weights have unit maximum magnitude. ``n`` is static because it fixes the
    output shape and recurrence length.

    The MATLAB top-level function handles n=0 and n=1 before method dispatch;
    these source cases are reproduced here. This helper supports n>=21, the
    source's default REC range and the range used by the positive-half seed
    splice. The public MATLAB default uses GW for n<=20. Its explicit small-n
    REC path interacts with the source's ten-entry Airy-root assignment to a
    shorter vector; that legacy corner is intentionally left unsupported
    here rather than replacing it with guessed seeds.
    """
    if n < 0:
        raise ValueError("hermpts_rec: n must be nonnegative")
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return empty, empty, empty
    if n == 1:
        return (jnp.zeros((1,), dtype=jnp.float64),
                jnp.full((1,), jnp.sqrt(jnp.pi), dtype=jnp.float64),
                jnp.ones((1,), dtype=jnp.float64))
    if n < 21:
        raise ValueError("hermpts_rec: REC is supported for n >= 21")

    x0 = _initial_guesses(n) * jnp.sqrt(2.0)
    dx0 = jnp.full_like(x0, jnp.inf)
    threshold = jnp.sqrt(jnp.finfo(jnp.float64).eps)

    def continue_newton(state):
        count, _x, dx = state
        return (count < 10) & (jnp.max(jnp.abs(dx)) >= threshold)

    def newton_step(state):
        count, x, _dx = state
        value, derivative = _hermite_scaled_value_derivative(n, x)
        dx = value / derivative
        dx = jnp.where(jnp.isnan(dx), 0.0, dx)
        return count + 1, x - dx, dx

    _, x_pos_scaled, _ = lax.while_loop(
        continue_newton, newton_step,
        (jnp.asarray(0, dtype=jnp.int32), x0, dx0),
    )
    _, derivative = _hermite_scaled_value_derivative(n, x_pos_scaled)
    x_pos = x_pos_scaled / jnp.sqrt(2.0)
    w_pos = jnp.exp(-(x_pos**2)) / derivative**2
    v_pos = jnp.exp(-(x_pos**2) / 2.0) / derivative

    if n % 2:
        x = jnp.concatenate((-x_pos[::-1], x_pos[1:]))
        w = jnp.concatenate((w_pos[::-1], w_pos[1:]))
        v = jnp.concatenate((v_pos[::-1], v_pos[1:]))
    else:
        x = jnp.concatenate((-x_pos[::-1], x_pos))
        w = jnp.concatenate((w_pos[::-1], w_pos))
        v = jnp.concatenate((v_pos[::-1], -v_pos))

    w = w * (jnp.sqrt(jnp.pi) / jnp.sum(w))
    v = v / jnp.max(jnp.abs(v))
    return x, w, v
