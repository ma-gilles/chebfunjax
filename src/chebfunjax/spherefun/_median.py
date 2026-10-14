"""Source tensor-product median for Spherefun.

Provenance
----------
MATLAB source: @spherefun/median.m, @separableApprox/median.m,
@chebfun/simplify.m; Chebfun7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref


def _odd_grid_median(values, axis):
    """MATLAB median on the odd source grid, including complex ordering.

    MATLAB numeric median sorts complex values by magnitude, then phase.
    The source Spherefun grid always has 2049 samples in each direction.
    """
    if not jnp.iscomplexobj(values):
        return jnp.median(values, axis=axis)
    if values.shape[axis] % 2 != 1:
        raise ValueError("The source median grid must have odd length")
    order = jnp.lexsort((jnp.angle(values), jnp.abs(values)), axis=axis)
    ordered = jnp.take_along_axis(values, order, axis=axis)
    result = jnp.take(ordered, values.shape[axis] // 2, axis=axis)
    return jnp.where(jnp.any(jnp.isnan(values), axis=axis), jnp.nan, result)


def source_median(f, dim):
    """Preserve fixed-grid sampling and ordinary numeric Chebfun conversion."""
    dimension = int(dim)
    if dimension != dim or dimension < 1:
        raise ValueError("Median dimension must be a positive integer")
    values = f.sample(2049, 2049)
    # MATLAB leaves a matrix unchanged when reducing a higher singleton dim.
    medians = _odd_grid_median(values, axis=dimension - 1) if dimension <= 2 else values
    interval = (-float(jnp.pi), float(jnp.pi)) if dimension == 1 else (0., float(jnp.pi))
    curve = chebfun(medians.reshape(-1, order="F"), domain=interval)
    # This is a single column, so globaltol and ordinary global vertical-scale
    # simplification agree. Read the current source preference explicitly.
    return curve.simplify(float(ChebfunPref().techPrefs.chebfuneps))
