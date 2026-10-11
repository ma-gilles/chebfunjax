"""Pinned @chebfun/permute.m validation and first-index dispatch.

Numeric array orders are a Python adapter to MATLAB numeric orders. Captured
validation identifiers are preserved; unsupported Python object types are not
claimed to reproduce every MATLAB builtin validation priority.
"""

import jax.numpy as jnp


def source_transposes(order):
    values = jnp.asarray(order)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise ValueError("MATLAB:permute:complexIndex: order must be real")
    flat = values.reshape(-1)
    if flat.size < 2:
        raise ValueError("MATLAB:permute:tooFewIndices: at least two indices required")
    if not bool(
        jnp.all(jnp.isfinite(flat) & (flat >= 1) & (flat <= flat.size) & (flat == jnp.floor(flat)))
    ):
        raise ValueError("MATLAB:permute:badIndex: invalid permutation index")
    if not bool(jnp.array_equal(jnp.sort(flat), jnp.arange(1, flat.size + 1))):
        raise ValueError("MATLAB:permute:repeatedIndex: repeated permutation index")
    # Literal source rule, including valid trailing singleton dimensions.
    return bool(flat[0] == 2)
