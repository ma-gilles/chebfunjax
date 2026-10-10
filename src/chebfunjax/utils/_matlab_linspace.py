"""Finite scalar grid adapter for the R2025b Needle source coordinates.

Provenance: MATLAB R2025b toolbox/matlab/elmat/linspace.m.
This is an eager binary64 adapter; the R2017a histogram adapter is unchanged.
"""
import operator

import jax.numpy as jnp

from chebfunjax.utils.matlab_hist import _source_divide, _source_linspace


def source_grid(left, right, count):
    """Construct a finite scalar binary64 grid using source arithmetic.

    Provenance
    ----------
    MATLAB builtin source: R2025b toolbox/matlab/elmat/linspace.m.
    Chebfun commit: 7574c77 (reference library; caller opt/Needle.m).
    """
    count = operator.index(count)
    left = jnp.asarray(left, dtype=jnp.float64)
    right = jnp.asarray(right, dtype=jnp.float64)
    if left.ndim or right.ndim or not bool(jnp.isfinite(left) & jnp.isfinite(right)):
        raise ValueError("finite scalar endpoints required")
    if count < 2:
        raise ValueError("at least two grid points required")
    intervals = count - 1
    if count > 2 and bool(left == -right):
        step = _source_divide(right, intervals)
        values = jnp.arange(-intervals, intervals + 1, 2, dtype=jnp.float64) * step
        values = values.at[0].set(left).at[-1].set(right)
        if intervals % 2 == 0:
            values = values.at[intervals // 2].set(0.)
        return values
    return _source_linspace(left, right, intervals)
