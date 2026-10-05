"""Pure-JAX Ai/Ai-prime on the two rays used by Laguerre RH.

This is a bounded numerical replacement for MATLAB's builtin airy calls,
not a port of MATLAB's builtin internals or a general complex Airy API.
The supported line is z=t*exp(-i*pi/3), t real. Invalid arguments return NaN.
Local evaluation propagates y''=z*y along quarter-unit panels; outside radius
16 it uses the 20-term complex asymptotic expansion with principal powers.

Provenance
----------
MATLAB source : lagpts.m (asyAiry, builtin airy(0,fn)/airy(1,fn))
Chebfun commit: 7574c77
ODE/initial values: https://dlmf.nist.gov/9.2.E1, .E3, .E4
Complex expansions/coefficient recurrence: https://dlmf.nist.gov/9.7.E5,
    https://dlmf.nist.gov/9.7.E6, https://dlmf.nist.gov/9.7.E2
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_RAY = complex(0.5, -0.86602540378443864676)


@jax.jit
def _airy_ray_tables():
    """Complex ODE Taylor tables on both directions from the origin."""
    degree = 31
    initial = (jnp.asarray(0.35502805388781723926, jnp.complex128),
               jnp.asarray(-0.25881940379280679841, jnp.complex128))

    def table(direction):
        step = direction * _RAY * 0.25

        def panel(state, index):
            y, dy = state
            center = index * step
            delta = (index + 1) * step - center
            coefficients = jnp.zeros((degree + 1,), dtype=jnp.complex128)
            coefficients = coefficients.at[0].set(y).at[1].set(dy)
            coefficients = coefficients.at[2].set(center * y / 2)

            def next_coefficient(k, values):
                value = (center * values[k] + values[k - 1]) / ((k + 1) * (k + 2))
                return values.at[k + 2].set(value)

            coefficients = lax.fori_loop(1, degree - 1, next_coefficient, coefficients)
            derivative = coefficients[1:] * jnp.arange(1, degree + 1)
            next_y = jnp.polyval(coefficients[::-1], delta)
            next_dy = jnp.polyval(derivative[::-1], delta)
            return (next_y, next_dy), coefficients

        return lax.scan(panel, initial, jnp.arange(64))[1]

    return jax.vmap(table)(jnp.asarray([1.0, -1.0]))


@jax.jit
def _airy_laguerre_rays(z):
    """Return Ai(z), Ai-prime(z) on the line through exp(-i*pi/3).

    Scalar and array values are supported under JIT and vmap. This bounded
    helper does not promise accuracy on other rays. Far-branch phase errors
    scale with |z|^(3/2); accuracy must be assessed using that conditioning.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    radius = jnp.abs(z)
    rotated = z * jnp.conj(jnp.asarray(_RAY))
    negative = jnp.real(rotated) < 0
    direction = jnp.where(negative, -1.0, 1.0)
    local_z = jnp.where(radius <= 16, z, direction * 16 * _RAY)
    index = jnp.minimum(jnp.floor(4 * jnp.minimum(radius, 16)).astype(jnp.int32), 63)
    coefficients = _airy_ray_tables()[negative.astype(jnp.int32), index]
    delta = local_z - direction * index * (0.25 * _RAY)

    def horner(value, coefficient):
        return value * delta + coefficient, None

    local_ai = lax.scan(horner, jnp.zeros_like(z),
                        jnp.moveaxis(coefficients[..., ::-1], -1, 0))[0]
    derivative = coefficients[..., 1:] * jnp.arange(1, 32)
    local_aip = lax.scan(horner, jnp.zeros_like(z),
                         jnp.moveaxis(derivative[..., ::-1], -1, 0))[0]

    # Keep the inactive asymptotic branch regular at z=0 for autodiff.
    far_z = jnp.where(radius > 16, z, 16 * _RAY)
    zeta = (2.0 / 3.0) * far_z * jnp.sqrt(far_z)
    initial = (jnp.ones_like(z), jnp.ones_like(z), jnp.ones_like(z))

    def term(k, state):
        value, u_sum, v_sum = state
        value = -value * ((6.0 * k - 5) * (6.0 * k - 1) / (72.0 * k)) / zeta
        v_value = value * ((6.0 * k + 1) / (1.0 - 6.0 * k))
        return value, u_sum + value, v_sum + v_value

    _, u_sum, v_sum = lax.fori_loop(1, 20, term, initial)
    quarter_power = jnp.sqrt(jnp.sqrt(far_z))
    exponential = jnp.exp(-zeta) / (2 * jnp.sqrt(jnp.pi))
    far_ai = exponential * u_sum / quarter_power
    far_aip = -exponential * quarter_power * v_sum
    ai = jnp.where(radius <= 16, local_ai, far_ai)
    aip = jnp.where(radius <= 16, local_aip, far_aip)
    eps = jnp.finfo(jnp.float64).eps
    valid = (jnp.isfinite(jnp.real(z)) & jnp.isfinite(jnp.imag(z))
             & (jnp.abs(jnp.imag(rotated)) <= 16 * eps * jnp.maximum(1, radius)))
    invalid = jnp.asarray(complex(float('nan'), float('nan')), dtype=z.dtype)
    return jnp.where(valid, ai, invalid), jnp.where(valid, aip, invalid)
