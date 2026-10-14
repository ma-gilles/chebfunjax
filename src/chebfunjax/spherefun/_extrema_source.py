"""Native real nonempty Spherefun extrema front-end.

Provenance: @spherefun/minandmax2.m, @spherefun/cdr.m,
@chebfun2/outerProduct.m; Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
"""
import jax.numpy as jnp

from chebfunjax.chebfun2d._active_set_source import native_active_set
from chebfunjax.chebfun2d._extrema_fallback import original_objective, refine_seed
from chebfunjax.chebfun2d._extrema_source import _iszero, _rank_one, _scale_factors
from chebfunjax.chebfun2d._svd import _as_fun, _axis_panel
from chebfunjax.utils.quadrature import trigpts


def _reconstruct(f):
    """Source C*D*R.' leaves the Chebfun2 outer product with unit pivots."""
    c, d, r = f.cdr()
    interval = (-float(jnp.pi), float(jnp.pi))
    cols = _as_fun(c[0] if len(c) == 1 else _axis_panel(c), interval) @ d
    rows = _as_fun(r[0] if len(r) == 1 else _axis_panel(r), interval)
    return cols @ rows.T, rows, cols


def _seed(rows, cols, objective):
    """Original trig lengths, appended pi, first column-major extrema."""
    xpts = jnp.concatenate((trigpts(len(rows), (rows.domain.a, rows.domain.b))[0],
                            jnp.asarray([jnp.pi])))
    ypts = jnp.concatenate((trigpts(len(cols), (cols.domain.a, cols.domain.b))[0],
                            jnp.asarray([jnp.pi])))
    cvals = jnp.asarray(cols(ypts)).reshape((ypts.size, -1))
    rvals = jnp.asarray(rows(xpts)).reshape((xpts.size, -1))
    flat = (cvals @ rvals.T).reshape(-1, order="F")
    indices = jnp.stack((jnp.argmin(flat), jnp.argmax(flat)))
    locations = jnp.stack((xpts[indices // ypts.size],
                           ypts[indices % ypts.size]), axis=1)
    values = jnp.stack((objective(locations[0]), objective(locations[1])))
    return values, locations


def source_extrema(sphere):
    """Literal real nonempty source branches; retain the native BMC map."""
    f, rows, cols = _reconstruct(sphere)
    if _iszero(f.approx):
        a, b, c, d = f.approx.domain
        center = jnp.asarray(((b+a)/2, (d+c)/2))
        return jnp.zeros((2,)), jnp.stack((center, center))
    rank = f.approx.rank
    rows, cols = _scale_factors(rows, cols, f.approx.pivots,
                                jnp.ones((rank,)))
    if rank == 1:
        values, locations = _rank_one(rows, cols)
    elif rank <= 4000:
        objective = original_objective(f.approx)
        values, locations = _seed(rows, cols, objective)
        values, locations = refine_seed(objective, values, locations,
                                         f.approx.domain,
                                         active_solver=native_active_set)
    else:
        raise ValueError("CHEBFUN:SPHEREFUN:minandmax2:length: Rank is too large.")
    negative = locations[:, 0] < 0
    locations = locations.at[:, 0].set(jnp.where(negative,
                                       locations[:, 0] + jnp.pi, locations[:, 0]))
    locations = locations.at[:, 1].set(jnp.where(negative,
                                       -locations[:, 1], locations[:, 1]))
    return values, locations
