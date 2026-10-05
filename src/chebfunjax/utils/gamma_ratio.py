"""Accurate gamma ratios for large base arguments.

This is a private, pure-JAX port of Chebfun's ``gammaratio.m``. Inputs are
scalar JAX values, so both ``m`` and ``delta`` may be dynamic arguments of a
JIT-compiled caller.

Provenance
----------
MATLAB source : gammaratio.m
Chebfun commit: 7574c77
Original authors: Nick Hale and Alex Townsend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

_STIRLING = (
    1.0,
    1.0 / 12.0,
    1.0 / 288.0,
    -139.0 / 51840.0,
    -571.0 / 2488320.0,
    163879.0 / 209018880.0,
    5246819.0 / 75246796800.0,
    -534703531.0 / 902961561600.0,
    -4483131259.0 / 86684309913600.0,
    432261921612371.0 / 514904800886784000.0,
)


def _stirling_correction(z):
    inverse_powers = jnp.concatenate((
        jnp.ones((1,), dtype=z.dtype),
        jnp.cumprod(jnp.ones((9,), dtype=z.dtype) / z),
    ))
    # A module-level JAX array can capture a tracer if this module is first
    # imported inside another jitted method (e.g. Hermite's LAG branch).
    return jnp.sum(jnp.asarray(_STIRLING, dtype=z.dtype) * inverse_powers)


def _taylor_ratio(m, delta):
    """MATLAB's Taylor/Stirling branch for ``m > 15`` and ``delta <= m``."""
    fd = jnp.floor(delta).astype(jnp.int32)
    rd = delta - fd.astype(delta.dtype)

    def strip_integer(_):
        def multiply(k, scale):
            return scale * (m + k.astype(m.dtype) + rd)

        scale = lax.fori_loop(0, fd, multiply, jnp.asarray(1.0, m.dtype))
        fractional = lax.cond(rd == 0.0, lambda __: jnp.asarray(1.0, m.dtype),
                              lambda __: _fractional_ratio(m, rd), operand=None)
        return scale * fractional

    def no_strip(_):
        return _fractional_ratio(m, delta)

    return lax.cond(fd > 1, strip_integer, no_strip, operand=None)


def _fractional_ratio(m, delta):
    """Source Taylor expansion for the fractional or direct delta."""
    eps = jnp.finfo(m.dtype).eps
    ds0 = 0.5 * delta**2 / (m - 1.0)
    init = (jnp.asarray(1, jnp.int32), ds0, ds0)

    def continue_series(state):
        j, ds, total = state
        relative = jnp.abs(ds / total)
        return (relative > eps / 100.0) & (j < 100)

    def add_term(state):
        j, ds, total = state
        j_next = j + 1
        ds_next = (-delta * (j_next - 1).astype(m.dtype)
                   / (j_next + 1).astype(m.dtype) / (m - 1.0) * ds)
        return j_next, ds_next, total + ds_next

    _, _, series = lax.while_loop(continue_series, add_term, init)
    p2 = (jnp.exp(series) * jnp.sqrt(1.0 + delta / (m - 1.0))
          * (m - 1.0)**delta)
    return p2 * (_stirling_correction(m + delta - 1.0)
                 / _stirling_correction(m - 1.0))


@jax.jit
def _gamma_ratio(m, delta):
    """Compute ``Gamma(m + delta) / Gamma(m)`` using Chebfun's algorithm.

    The small-argument fallback and the integer-delta branch follow the
    MATLAB source literally, including its ``floor(delta) > 1`` condition.
    ``m`` and ``delta`` must be scalar real values.

    Provenance
    ----------
    MATLAB source : gammaratio.m
    Chebfun commit: 7574c77
    """
    m = jnp.asarray(m, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    if m.ndim != 0 or delta.ndim != 0:
        raise ValueError("_gamma_ratio expects scalar m and delta")

    def fallback(_):
        return jnp.exp(gammaln(m + delta) - gammaln(m))

    def large_base(_):
        return lax.cond(
            delta == 0.0,
            lambda __: jnp.asarray(1.0, dtype=m.dtype),
            lambda __: _taylor_ratio(m, delta),
            operand=None,
        )

    return lax.cond((m <= 15.0) | (m < delta), fallback, large_base,
                    operand=None)
