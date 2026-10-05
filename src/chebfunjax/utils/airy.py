"""Pure-JAX real-negative Airy evaluation for Hermite asymptotics.

MATLAB's ``hermpts.m`` calls its built-in ``airy``; JAX has no equivalent.
On [-16, 0] we propagate the defining ODE in quarter-unit Taylor panels.
Farther left we use the oscillatory asymptotic expansion, keeping 20 terms.
This helper is deliberately limited to the real-negative arguments used by
the Hermite ASY algorithm, rather than presenting an incomplete public Airy API.

Provenance
----------
MATLAB source : hermpts.m (``hermpoly_asy_airy``, built-in ``airy`` calls)
Chebfun commit: 7574c77
Airy ODE and initial values: https://dlmf.nist.gov/9.2.E1,
    https://dlmf.nist.gov/9.2.E3, https://dlmf.nist.gov/9.2.E4
Oscillatory expansions: https://dlmf.nist.gov/9.7.E9,
    https://dlmf.nist.gov/9.7.E10; coefficient recurrence 9.7.E2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def _airy_taylor_table():
    """Taylor coefficients at 0, -1/4, ..., -63/4 from y'' = x*y."""
    # Correctly rounded binary64 initial values of the DLMF gamma formulas.
    initial = (jnp.asarray(0.35502805388781723926, dtype=jnp.float64),
               jnp.asarray(-0.25881940379280679841, dtype=jnp.float64))
    degree = 31

    def panel(state, center):
        y, dy = state
        coefficients = jnp.zeros((degree + 1,), dtype=jnp.float64)
        coefficients = coefficients.at[0].set(y).at[1].set(dy)
        coefficients = coefficients.at[2].set(center * y / 2.0)

        def next_coefficient(k, a):
            return a.at[k + 2].set(
                (center * a[k] + a[k - 1]) / ((k + 1) * (k + 2))
            )

        coefficients = lax.fori_loop(1, degree - 1, next_coefficient,
                                     coefficients)
        derivative = coefficients[1:] * jnp.arange(1, degree + 1)
        next_y = jnp.polyval(coefficients[::-1], -0.25)
        next_dy = jnp.polyval(derivative[::-1], -0.25)
        return (next_y, next_dy), coefficients

    _, table = lax.scan(panel, initial, -0.25 * jnp.arange(64))
    return table


@jax.jit
def _airy_negative(x):
    """Return Ai(x) and Ai'(x) for real x <= 0, using binary64 JAX arrays.

    Invalid positive arguments return NaN. On the oscillatory branch the
    absolute error grows with phase-rounding error, so a relative error
    specification at zeros would be inappropriate.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    # Preserve the ODE derivative at the endpoints; clip has a half-gradient
    # there, whereas the valid branch should retain the full derivative.
    local_x = jnp.where((x >= -16.0) & (x <= 0.0), x,
                        jnp.clip(x, -16.0, 0.0))
    index = jnp.minimum(jnp.floor(-4.0 * local_x).astype(jnp.int32), 63)
    delta = local_x + 0.25 * index
    a = _airy_taylor_table()[index]

    def horner(carry, coefficient):
        return carry * delta + coefficient, None

    local_ai, _ = lax.scan(horner, jnp.zeros_like(x), jnp.moveaxis(a[..., ::-1], -1, 0))
    derivative = a[..., 1:] * jnp.arange(1, 32)
    local_aip, _ = lax.scan(horner, jnp.zeros_like(x),
                            jnp.moveaxis(derivative[..., ::-1], -1, 0))

    t = jnp.maximum(-x, 16.0)
    zeta = (2.0 / 3.0) * t * jnp.sqrt(t)
    inverse_zeta = 1.0 / zeta
    # DLMF 9.7.E2: u_k/u_(k-1) = (6k-5)(6k-1)/(72k),
    # v_k = (6k+1)/(1-6k) u_k. Powers are accumulated without large u_k.
    initial = (jnp.ones_like(x), jnp.ones_like(x), jnp.zeros_like(x),
               jnp.ones_like(x), jnp.zeros_like(x))

    def asymptotic_term(k, state):
        term, ue, uo, ve, vo = state
        term = term * ((6.0 * k - 5.0) * (6.0 * k - 1.0) / (72.0 * k)) * inverse_zeta
        vterm = term * ((6.0 * k + 1.0) / (1.0 - 6.0 * k))
        sign = jnp.where((k // 2) % 2, -1.0, 1.0)
        even = k % 2 == 0
        return (term, ue + jnp.where(even, sign * term, 0.0),
                uo + jnp.where(even, 0.0, sign * term),
                ve + jnp.where(even, sign * vterm, 0.0),
                vo + jnp.where(even, 0.0, sign * vterm))

    _, ue, uo, ve, vo = lax.fori_loop(1, 20, asymptotic_term, initial)
    phase = zeta - jnp.pi / 4.0
    sine, cosine = jnp.sin(phase), jnp.cos(phase)
    amplitude = t**0.25
    far_ai = (cosine * ue + sine * uo) / (jnp.sqrt(jnp.pi) * amplitude)
    far_aip = (sine * ve - cosine * vo) * amplitude / jnp.sqrt(jnp.pi)
    ai = jnp.where(x >= -16.0, local_ai, far_ai)
    aip = jnp.where(x >= -16.0, local_aip, far_aip)
    return jnp.where(x <= 0.0, ai, jnp.nan), jnp.where(x <= 0.0, aip, jnp.nan)
