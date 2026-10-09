"""Native Chebtech continuous minandmax policy, shared by C1 and C2.

Provenance
----------
MATLAB source: @chebtech/minandmax.m, @chebtech/isreal.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp

from chebfunjax.utils.quadrature import chebpts


def _extreme(values, *, maximum=False):
    """First real extremum, ignoring NaNs unless all values are NaN."""
    values = jnp.asarray(values)
    valid = ~jnp.isnan(values)
    fill = -jnp.inf if maximum else jnp.inf
    keys = jnp.where(valid, values, fill)
    target = jnp.max(keys) if maximum else jnp.min(keys)
    index = jnp.argmax(valid & (values == target))
    return values[index], index


def _column(f, fp, points):
    if f.n == 1:
        position = jnp.asarray(0., dtype=jnp.float64)
        value = jnp.real(f(position))
        return (value, position), (value, position)
    roots = fp.roots()
    candidates = jnp.concatenate((jnp.asarray([-1.]), roots,
                                  jnp.asarray([1.])))
    values = jnp.real(f(candidates))
    minimum, imin = _extreme(values)
    xmin = candidates[imin]
    grid_values = jnp.real(f.coeffs2vals(f.coeffs))
    vmin, index = _extreme(jnp.concatenate((minimum[None], grid_values)))
    if bool(vmin < minimum):
        minimum, xmin = vmin, points[index-1]
    maximum, imax = _extreme(values, maximum=True)
    xmax = candidates[imax]
    # Literal pinned source uses MIN here, not MAX. For finite values this
    # cannot improve maximum. Preserve the source quirk instead of repairing it.
    vmax, index = _extreme(jnp.concatenate((maximum[None], grid_values)))
    if bool(vmax > maximum):
        maximum, xmax = vmax, points[index-1]
    return (minimum, xmin), (maximum, xmax)


def minandmax(f, *, kind):
    """Return ((minimum, location), (maximum, location)); roots remain eager."""
    if jnp.iscomplexobj(f.coeffs):
        realf, imagf = f.real(), f.imag()
        squared = (realf*realf + imagf*imagf).simplify()
        (_, xmin), (_, xmax) = minandmax(squared, kind=kind)
        if f.coeffs.ndim == 2:
            vmin = jnp.diagonal(f(jnp.atleast_1d(xmin)))
            vmax = jnp.diagonal(f(jnp.atleast_1d(xmax)))
        else:
            vmin, vmax = f(xmin), f(xmax)
        return (vmin, xmin), (vmax, xmax)
    derivative = f.diff()
    points = chebpts(f.n, kind=kind)
    if f.coeffs.ndim == 1:
        return _column(f, derivative, points)
    if f.coeffs.shape[1] == 0:
        # Native initializes 2-by-numColumns values/positions and executes
        # zero column iterations; adapt those rows to the Python tuple API.
        empty = jnp.empty((0,), dtype=jnp.float64)
        return (empty, empty), (empty, empty)
    cls = type(f)
    output = [_column(cls(coeffs=f.coeffs[:, j], ishappy=f.ishappy),
                      cls(coeffs=derivative.coeffs[:, j],
                          ishappy=derivative.ishappy), points)
              for j in range(f.coeffs.shape[1])]
    return ((jnp.stack([r[0][0] for r in output]),
             jnp.stack([r[0][1] for r in output])),
            (jnp.stack([r[1][0] for r in output]),
             jnp.stack([r[1][1] for r in output])))
