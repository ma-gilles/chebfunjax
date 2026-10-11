"""Stored point division at exactly zero denominators.

Provenance
----------
MATLAB source: @chebfun/rdivide.m, lines 195–204 and 218–219.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
MATLAB zero-denominator values are qualified separately from finite division.
"""

import jax
import jax.numpy as jnp


def divide_point_values(numerator, denominator):
    """Preserve native scalar zero-division values with complex storage."""
    numerator, denominator = jnp.asarray(numerator), jnp.asarray(denominator)
    zero = jnp.real(denominator)
    if jnp.iscomplexobj(numerator):
        re, im = jnp.real(numerator), jnp.imag(numerator)
        re = jnp.where(re == 0, 0.0, re / zero)
        im = jnp.where(im == 0, 0.0, im / zero)
        at_zero = jax.lax.complex(re, im)
        at_zero = jnp.where(
            numerator == 0, jax.lax.complex(jnp.full_like(re, jnp.nan), jnp.zeros_like(im)), at_zero
        )
    else:
        at_zero = numerator / zero
    return jnp.where(denominator == 0, at_zero, numerator / denominator)
