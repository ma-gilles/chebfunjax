"""Source real separable extrema front-end and rank-one policy.

Higher ranks use the qualified source fallback because no source-equivalent
active-set optimizer is implemented. This does not establish parity with native
active-set installations. Complex extrema return None for the legacy adapter. Fixed4000
reconstruction uses the public constructor and its selected session technology.

Provenance
----------
MATLAB source: @separableApprox/{minandmax2,iszero,fevalm,cdr}.m,
    @chebfun/chebfun.m (parseOp), @chebtech/iszero.m, @trigtech/iszero.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp

from chebfunjax.chebfun2d._cdr_source import _mesh_values, _slice_values
from chebfunjax.chebfun2d._extrema_fallback import source_higher_extrema
from chebfunjax.chebfun2d._pivot_metadata import _retained_pivots
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


def _slice_iszero(piece):
    """Exact native technology predicate; NaNs compare unequal to zero."""
    if isinstance(piece, Trigtech):
        data = piece.values
    elif isinstance(piece, (Chebtech1, Chebtech2)):
        data = piece.coeffs
    else:
        raise NotImplementedError("Source extrema zero check requires polynomial or Trigtech slices")
    return bool(jnp.all(data == 0))


def _iszero(approx):
    """Literal reciprocal-pivot shortcut, 10x10 mesh, then slice check."""
    raw = _retained_pivots(approx)
    reciprocal = jnp.asarray(approx.pivots) if raw is None else 1/raw
    if bool(jnp.all(reciprocal == 0)):
        return True
    xa, xb, ya, yb = approx.domain
    values = _mesh_values(approx, jnp.linspace(xa, xb, 10),
                          jnp.linspace(ya, yb, 10))
    if bool(jnp.linalg.norm(values, ord=jnp.inf) > 0):
        return False
    cols_zero, rows_zero = [], []
    for col, row in zip(approx.cols, approx.rows):
        cols_zero.append(_slice_iszero(col))
        rows_zero.append(_slice_iszero(row))
    return all(cols_zero) or all(rows_zero)


def _isreal(approx):
    if jnp.iscomplexobj(approx.pivots):
        return False
    for piece in (*approx.cols, *approx.rows):
        if isinstance(piece, Trigtech):
            if not piece.is_real:
                return False
        elif jnp.iscomplexobj(piece.coeffs):
            return False
    return True


def _reconstruct(slices, a, b):
    """Native parseOp callback conversion, fixed4000, then simplify.

    The callback is intentional: passing an existing periodic Chebfun to the
    current factory can preserve its old technology instead of selected prefs.
    No private subset of constructor preference resolution is installed here.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    result = chebfun(lambda x: _slice_values(slices, x, a, b),
                     domain=(a, b), n=4000)
    # Native simplify consults the session tolerance. Pass it explicitly also
    # for the scalar-column adapter, whose omitted tolerance remains separate.
    return result.simplify(float(ChebfunPref().techPrefs.chebfuneps))


def _scale_factors(rows, cols, weights, pivot_values=None):
    """Use retained native pivots, or the explicitly limited legacy recovery.

    Do not substitute sqrt(abs(weights)): its rounding differs from the source.
    """
    pivots = 1 / jnp.asarray(weights) if pivot_values is None else pivot_values
    sq = 1 / jnp.sqrt(jnp.abs(pivots))
    rows = rows @ jnp.diag(sq * jnp.sign(pivots))
    cols = cols @ jnp.diag(sq)
    return rows, cols


def _rank_one(rows, cols):
    """Four source products of already-scaled continuous factor extrema."""
    (xr0, yr0), (xr1, yr1) = rows.minandmax()
    (xc0, yc0), (xc1, yc1) = cols.minandmax()
    yr0, yr1, yc0, yc1 = (jnp.asarray(v).reshape(())
                          for v in (yr0, yr1, yc0, yc1))
    xr0, xr1, xc0, xc1 = (jnp.asarray(v).reshape(())
                          for v in (xr0, xr1, xc0, xc1))
    products = jnp.stack((yr0*yc0, yr0*yc1, yr1*yc0, yr1*yc1))
    indices = jnp.stack((jnp.argmin(products), jnp.argmax(products)))
    locations = jnp.stack((jnp.where(indices < 2, xr0, xr1),
                           jnp.where(indices % 2 == 0, xc0, xc1)), axis=1)
    return products[indices], locations


def source_extrema(f):
    """Public real branch with explicit absence of active-set capability."""
    if f.isempty():
        return jnp.empty((0,), dtype=jnp.float64), jnp.empty((0,), dtype=jnp.float64)
    approx = f.approx
    if _iszero(approx):
        xa, xb, ya, yb = approx.domain
        center = jnp.asarray(((xb + xa)/2, (yb + ya)/2))
        return jnp.zeros((2,)), jnp.stack((center, center))
    if not _isreal(approx):
        return None
    xa, xb, ya, yb = approx.domain
    rows = _reconstruct(approx.rows, xa, xb)
    cols = _reconstruct(approx.cols, ya, yb)
    raw = _retained_pivots(approx)
    if raw is None:
        rows, cols = _scale_factors(rows, cols, approx.pivots)
    else:
        rows, cols = _scale_factors(rows, cols, approx.pivots, raw)
    if approx.rank == 1:
        return _rank_one(rows, cols)
    # No source-equivalent active-set implementation exists in this library.
    # None selects the source fallback transparently, without a fake failure.
    return source_higher_extrema(approx, rows, cols, active_solver=None)
