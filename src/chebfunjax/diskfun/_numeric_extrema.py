"""Native numeric Diskfun absolute extrema without float comparison FTZ.

Chebfun7574c77 @diskfun/constructor.m uses max/min with missing NaNs
omitted and first column-major maximum indices. Positive binary64 magnitude
bits have the same ordering as their values, including subnormal values.
This helper changes no arithmetic provider or NaN payload policy globally.
"""

import jax
import jax.numpy as jnp


@jax.jit
def source_abs_extrema(values):
    """Return(minabs,maxabs,firstmaxindex) for a nonempty binary64 array.

    Indices refer to Fortran flattening. All-NaN input retains its first
    absolute NaN and index0; NaNs do not win ties with valid zero entries.
    """
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1, order="F")
    if values.size == 0:
        raise ValueError("source_abs_extrema requires a nonempty array")
    bits = jax.lax.bitcast_convert_type(values, jnp.uint64) & jnp.uint64(0x7FFFFFFFFFFFFFFF)
    valid = bits <= jnp.uint64(0x7FF0000000000000)
    present = jnp.any(valid)
    maximum = jnp.max(jnp.where(valid, bits, jnp.uint64(0)))
    minimum = jnp.min(jnp.where(valid, bits, jnp.uint64(0xFFFFFFFFFFFFFFFF)))
    index = jnp.argmax(valid & (bits == maximum))
    maximum = jnp.where(present, maximum, bits[0])
    minimum = jnp.where(present, minimum, bits[0])
    index = jnp.where(present, index, 0)
    return (
        jax.lax.bitcast_convert_type(minimum, jnp.float64),
        jax.lax.bitcast_convert_type(maximum, jnp.float64),
        index,
    )
