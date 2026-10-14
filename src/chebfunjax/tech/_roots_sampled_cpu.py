"""Reuse CPU sampled-subdivision grid and recurrence compilation shapes.

Provenance
----------
MATLAB source: @chebtech/roots.m, @chebtech/clenshaw.m, @chebtech2/chebpts.m.
Chebfun commit: 7574c77.
The binary64 division body is the existing independently derived integer
quotient/remainder implementation in utils/_binary64.py. Only host integer
normalization metadata becomes dynamic. Its global API remains unchanged.
The existing fdlibm sine sequence and original-n final FFT are retained.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import _vals2coeffs_jax
from chebfunjax.utils._chebpts_cpu import divide_normalized as divide_normalized
from chebfunjax.utils._fdlibm_sine import _sin_cheb_angles


@jax.jit
def affine_points(x, a, b):
    # Preserve the original mapped-points helper's dynamic endpoint boundary.
    return b*(x + 1)/2 + a*(1 - x)/2

@partial(jax.jit, static_argnames=("capacity",))
def dynamic_points(n, k, d, capacity):
    i = jnp.arange(capacity, dtype=jnp.int64)
    m = n - 1
    numerator = (2*i - m).astype(jnp.float64)
    numerator = jnp.where(i < n, numerator, 0.)
    angles = divide_normalized(jnp.pi*numerator, k, d)
    x = _sin_cheb_angles(angles)
    left = affine_points(x, -1., -0.004849834917525)
    right = affine_points(x, -0.004849834917525, 1.)
    j = jnp.arange(2*capacity, dtype=jnp.int64)
    points = jnp.where(j < n, left[jnp.minimum(j, capacity-1)],
                      right[jnp.clip(j - n, 0, capacity-1)])
    return jnp.where(j < 2*n, points, 0.)


@jax.jit
def recurrence_states(coeffs, points, n):
    x2 = 2.0 * points
    degree = n - 1

    def body(i, state):
        (b1, b2) = state
        k = degree - 2 * i
        b2 = coeffs[k] + x2 * b1 - b2
        b1 = coeffs[k - 1] + x2 * b2 - b1
        return (b1, b2)
    (b1, b2) = jax.lax.fori_loop(0, degree // 2, body, (jnp.zeros_like(points), jnp.zeros_like(points)))
    return (b1, b2, jnp.all(jnp.isfinite(coeffs)))

@partial(jax.jit, static_argnames=('n',))
def finish_subdivision(coeffs, points, b1, b2, n):
    points = points[:2 * n]
    (b1, b2) = (b1[:2 * n], b2[:2 * n])
    if (n - 1) % 2:
        (b1, b2) = (coeffs[1] + 2.0 * points * b1 - b2, b1)
    values = coeffs[0] + points * b1 - b2
    return (_vals2coeffs_jax(values[:n]), _vals2coeffs_jax(values[n:]))

@jax.jit
def grid_and_states(coeffs, n, k, d):
    points = dynamic_points(n, k, d, coeffs.shape[0])
    b1, b2, finite = recurrence_states(coeffs, points, n)
    return points, b1, b2, finite
