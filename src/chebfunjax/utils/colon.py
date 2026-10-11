"""Finite double-precision source colon grids.

Provenance
----------
MATLAB R2025b toolbox/eml/lib/matlab/ops/colon.m, eml_float_colon and
float_colon_length. Runtime adapter for examples using Chebfun commit 7574c77.
"""

import math

import jax.numpy as jnp


def matlab_colon(start: float, step: float, stop: float) -> jnp.ndarray:
    """Return a finite scalar ``start:step:stop`` grid in float64.

    Endpoint/count decisions use concrete scalar inputs, so the output length
    must be known eagerly. Integer, single, character, nonfinite and traced
    input dispatch are outside this adapter. A zero step or reversed direction
    produces an empty grid, as in the source floating-point branch.

    Provenance
    ----------
    MATLAB R2025b toolbox/eml/lib/matlab/ops/colon.m:315-397, paired endpoint
    filling and floating-point length calculation. Chebfun commit: 7574c77.
    """
    a, d, b = float(start), float(step), float(stop)
    if not all(math.isfinite(v) for v in (a, d, b)):
        raise ValueError("matlab_colon requires finite scalar inputs")
    if d == 0 or (a < b and d < 0) or (b < a and d > 0):
        return jnp.empty((0,), dtype=jnp.float64)
    nearest = math.floor((b-a)/d + .5)
    endpoint = a + nearest*d
    difference = endpoint-b if d > 0 else b-endpoint
    threshold = 2*float(jnp.finfo(jnp.float64).eps)*max(abs(a), abs(b))
    if abs(difference) < threshold:
        n, endpoint = nearest+1, b
    elif difference > 0:
        n, endpoint = nearest, a+(nearest-1)*d
    else:
        n = nearest+1
    if n <= 0:
        return jnp.empty((0,), dtype=jnp.float64)
    if n == 1:
        return jnp.asarray([a], dtype=jnp.float64)
    k = jnp.arange(n, dtype=jnp.float64)
    # Compute each side from its own endpoint; do not generate the upper
    # half by accumulating or multiplying forward from the lower endpoint.
    lower = a + k*d
    upper = endpoint - (n-1-k)*d
    values = jnp.where(k < n//2, lower, upper)
    if n % 2:
        values = values.at[n//2].set((a+endpoint)/2)
    return values
