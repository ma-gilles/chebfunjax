"""Ordered real CPU Clenshaw evaluation, with ordinary JAX differentiation.

Provenance
----------
MATLAB source : @chebtech/clenshaw.m
Chebfun commit: 7574c77 (7574c77680d7e82b79626300bf255498271a72df).
The paired recurrence and final half-times-doubled argument follow that source.
Each binary operation uses the existing CPU ordering primitive; no host numerical
provider, degree padding, or input-specific coefficients are used. Native rounding
qualification covers finite ordinary inputs; exceptional/subnormal behavior is
not claimed bitwise native from that corpus.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary


def source_clenshaw(coeffs, x):
    """Evaluate real binary64 arrays; caller handles empty/constant series."""
    if coeffs.ndim == 2:
        x = x[..., None]
    x2 = binary(jnp.asarray(2.0, x.dtype), x, 0)
    shape = jnp.broadcast_shapes(x.shape, coeffs.shape[1:])
    zero = jnp.zeros(shape, dtype=coeffs.dtype)
    degree = coeffs.shape[0] - 1

    def step(i, state):
        b1, b2 = state
        k = degree - 2 * i
        next2 = binary(binary(coeffs[k], binary(x2, b1, 0), 1), b2, 2)
        next1 = binary(binary(coeffs[k - 1], binary(x2, next2, 0), 1), b1, 2)
        return next1, next2

    b1, b2 = jax.lax.fori_loop(0, degree // 2, step, (zero, zero))
    if degree % 2:
        b1, b2 = binary(binary(coeffs[1], binary(x2, b1, 0), 1), b2, 2), b1
    half_x = binary(jnp.asarray(0.5, x.dtype), x2, 0)
    return binary(binary(coeffs[0], binary(half_x, b1, 0), 1), b2, 2)
