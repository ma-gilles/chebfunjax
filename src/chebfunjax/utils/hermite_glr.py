"""Linear-storage Glaser--Liu--Rokhlin Gauss--Hermite quadrature.

Provenance
----------
MATLAB source : hermpts.m (alg0_Herm, alg1_Herm, alg2_Herm, rk2_Herm)
Chebfun commit: 7574c77
Original author: Nick Hale, March 2010.
"""

from functools import partial

import jax
import jax.numpy as jnp


def _phase_step(x, n, start):
    """Ten second-order Runge--Kutta steps for the Hermite phase ODE."""
    step = (-jnp.pi / 2 - start) / 10

    def advance(i, value):
        phase = start + i * step
        q = 2 * n + 1 - value * value
        first = -step / (jnp.sqrt(q) - value * jnp.sin(2 * phase) / (2 * q))
        next_value = value + first
        q = 2 * n + 1 - next_value * next_value
        second = -step / (jnp.sqrt(q)
                          - next_value * jnp.sin(2 * (phase + step)) / (2 * q))
        return value + (first + second) / 2

    return jax.lax.fori_loop(0, 10, advance, x)


def _root_from_taylor(x, value, derivative, spacing, n):
    """Refine the next root using 30 scaled Taylor coefficients."""
    c1 = -(2 * n + 1 - x * x) * spacing**2
    c2, c3 = 2 * x * spacing**3, spacing**4
    coefficients = jnp.zeros(31, dtype=jnp.float64)
    coefficients = coefficients.at[0].set(value).at[1].set(derivative * spacing)
    coefficients = coefficients.at[2].set(c1 * value / 2)
    coefficients = coefficients.at[3].set((c1 * coefficients[1] + c2 * value) / 6)

    def coefficient(k, c):
        return c.at[k + 2].set(
            (c1 * c[k] + c2 * c[k - 1] + c3 * c[k - 2]) / ((k + 1) * (k + 2)))

    coefficients = jax.lax.fori_loop(2, 29, coefficient, coefficients)
    differentiated = coefficients[1:] * jnp.arange(1, 31)

    def refine(_, z):
        val = jnp.polyval(coefficients[::-1], z)
        der = jnp.polyval(differentiated[::-1], z)
        return z - val / der

    z = jax.lax.fori_loop(0, 10, refine, jnp.asarray(1.0))
    return x + spacing * z, jnp.polyval(differentiated[::-1], z) / spacing


@partial(jax.jit, static_argnames=("n",))
def _hermpts_glr(n):
    """Hermite nodes, quadrature weights, and normalized barycentric weights."""
    def at_zero(k, state):
        hm2, hm1, hpm2, hpm1 = state
        ratio = jnp.sqrt(k / (k + 1.0))
        h = -ratio * hm2
        hp = jnp.sqrt(2 / (k + 1.0)) * hm1 - ratio * hpm2
        return hm1, h, hpm1, hp

    _, h, _, hp = jax.lax.fori_loop(
        0, n, at_zero, (0.0, jnp.pi**(-0.25), 0.0, 0.0))
    if n % 2:
        first = (jnp.asarray(0.0), hp)
        count = n // 2
    else:
        spacing = _phase_step(jnp.asarray(0.0), n, 0.0)
        first = _root_from_taylor(0.0, h, 0.0, spacing, n)
        count = n // 2 - 1

    def next_root(previous, _):
        x, derivative = previous
        spacing = _phase_step(x, n, jnp.pi / 2) - x
        result = _root_from_taylor(x, 0.0, derivative, spacing, n)
        return result, result

    _, (roots, derivatives) = jax.lax.scan(next_root, first, None, length=count)
    roots = jnp.concatenate([jnp.asarray(first[0])[None], roots])
    derivatives = jnp.concatenate([jnp.asarray(first[1])[None], derivatives])
    if n % 2:
        roots = jnp.concatenate([-roots[:0:-1], roots])
        derivatives = jnp.concatenate([derivatives[:0:-1], derivatives])
    else:
        roots = jnp.concatenate([-roots[::-1], roots])
        derivatives = jnp.concatenate([derivatives[::-1], derivatives])
    weights = 2 * jnp.exp(-roots * roots) / (derivatives * derivatives)
    weights = weights * (jnp.sqrt(jnp.pi) / jnp.sum(weights))
    bary = jnp.exp(-roots * roots / 2) / jnp.abs(derivatives)
    bary = bary / jnp.max(bary) * jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    return roots, weights, bary
