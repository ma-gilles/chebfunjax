"""JAX numeric getTol translation with source scalar power.

Source: @diskfun/constructor.m, Chebfun7574c77680d7e82b79626300bf255498271a72df.
Native getTol forms forward differences, divides by the grid spacings, and
scales the larger gradient/value magnitude by grid**(2/3), domain and eps.
The preexisting fdlibm scalar power kernel is qualified against native grid
powers; this is source compatibility, not a correctly rounded pow claim.
No native operands or values are cached.
"""

import jax.numpy as jnp

from chebfunjax.operators import _qz_ieee as ieee
from chebfunjax.utils._signed_gradual import flip_sign_bits
from chebfunjax.utils._source_real import source_pow_jit

from ._numeric_extrema import source_abs_extrema


def _gradients(values, hx, hy):
    """Source forward differences and individually rounded direct divisions."""
    m, n = values.shape
    dx = ieee.add(values[:m - 1, 1:], flip_sign_bits(values[:m - 1, :-1]))
    dy = ieee.add(values[1:, :n - 1], flip_sign_bits(values[:-1, :n - 1]))
    return ieee.divide(dx, hx), ieee.divide(dy, hy)


def numeric_tolerance(values, hx, hy, pseudo_level):
    """Source numeric tolerance and vscale as JAX binary64 arrays.

    Integer magnitude extrema retain subnormals and omit NaNs like source
    max. Only the maximum value is used, so concatenating the two gradient
    arrays preserves the source pairwise-max/reduction result. The empty
    gradient result is native0x1. Generic NaN payloads/AD are unqualified.
    """
    values = jnp.asarray(values, jnp.float64)
    m, n = values.shape
    hx, hy = jnp.asarray(hx, jnp.float64), jnp.asarray(hy, jnp.float64)
    dfdx, dfdy = _gradients(values, hx, hy)
    vscale = source_abs_extrema(values)[1]
    if dfdx.size == 0:
        return jnp.empty((0, 1), dtype=jnp.float64), vscale
    gradients = jnp.concatenate((dfdx.reshape(-1, order='F'),
                                 dfdy.reshape(-1, order='F')))
    jac_norm = source_abs_extrema(gradients)[1]
    scale = source_abs_extrema(jnp.stack((jac_norm, vscale)))[1]
    exponent = ieee.divide(jnp.float64(2), jnp.float64(3))
    # A nonempty array grid is an integer in [1, sys.maxsize]. Its binary64
    # value is positive normal and <=2**63 on this 64-bit runtime; y=2/3
    # gives |y*log2(grid)|<=42, inside the ordinary fdlibm kernel contract.
    # No cutoff tied to the captured grids or native buffers is involved.
    power = source_pow_jit(jnp.float64(max(m, n)), exponent)
    domain_scaled = ieee.multiply(power, jnp.float64(jnp.pi))
    value_scaled = ieee.multiply(domain_scaled, scale)
    tol = ieee.multiply(value_scaled, jnp.asarray(pseudo_level, jnp.float64))
    return tol, vscale
