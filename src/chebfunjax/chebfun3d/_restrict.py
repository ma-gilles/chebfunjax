"""Factor-preserving restriction of a Chebfun3.

Provenance
----------
MATLAB source: @chebfun3/restrict.m, Chebfun commit 7574c77.
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun1d.mtimes import _columns

from ._mtimes import _native_squeeze, _panel


def source_restrict(f, dom):
    """Restrict factors, using sequential native tensor contractions.

    Python real numeric sequences stand for MATLAB double domain vectors.
    Fixed coordinates are evaluated; free intervals use Chebfun.restrict.
    """
    try:
        domain = jnp.asarray(dom)
    except (TypeError, ValueError):
        raise ValueError('CHEBFUN:CHEBFUN3:restrict:domain: Unrecognizable domain.') from None
    if domain.dtype.kind not in 'iuf':
        raise ValueError('CHEBFUN:CHEBFUN3:restrict:domain: Unrecognizable domain.')
    if domain.size != 6:
        raise ValueError('CHEBFUN:CHEBFUN3:restrict:domain: Domain not determined.')
    d = tuple(float(x) for x in domain.reshape(-1))
    fixed = tuple(d[2*k] == d[2*k+1] for k in range(3))
    if all(fixed):
        return f(d[0], d[2], d[4])

    def panel(k):
        return _panel((f.cols, f.rows, f.tubes)[k], tuple(f.domain[2*k:2*k+2]))

    def restricted(k):
        return Chebfun.horzcat(*[c.restrict(d[2*k:2*k+2]) for c in _columns(panel(k))])

    def values(k):
        return jnp.asarray(panel(k)(d[2*k])).reshape((1, -1))

    def scalar_line(factors, core):
        # A native line has one function column. Expose the Python scalar
        # evaluation shape without changing coefficients or point values.
        return _columns(factors @ core)[0].extract_columns(0)

    if fixed == (False, True, True):
        cols = restricted(0)
        rows, tubes = values(1), values(2)
        core = f.txm(f.txm(f.core, rows, 2), tubes, 3)
        return scalar_line(cols, core.reshape((core.shape[0], 1)))
    if fixed == (True, False, True):
        cols, rows, tubes = values(0), restricted(1), values(2)
        core = f.txm(f.txm(f.core, cols, 1), tubes, 3)
        return scalar_line(rows, core.reshape((1, core.shape[1])).T)
    if fixed == (True, True, False):
        cols, rows, tubes = values(0), values(1), restricted(2)
        core = _native_squeeze(f.txm(f.txm(f.core, cols, 1), rows, 2))
        return scalar_line(tubes, core)
    if fixed == (True, False, False):
        cols, rows, tubes = values(0), restricted(1), restricted(2)
        core = _native_squeeze(f.txm(f.core, cols, 1))
        return (tubes @ core.T) @ rows.T
    if fixed == (False, True, False):
        cols, rows, tubes = restricted(0), values(1), restricted(2)
        core = _native_squeeze(f.txm(f.core, rows, 2))
        return (tubes @ core.T) @ cols.T
    if fixed == (False, False, True):
        cols, rows, tubes = restricted(0), restricted(1), values(2)
        core = f.txm(f.core, tubes, 3)
        return (rows @ core.reshape(core.shape[:2]).T) @ cols.T

    from .chebfun3 import Chebfun3
    factors = [_columns(restricted(k)) for k in range(3)]
    if any(len(c.funs) != 1 for mode in factors for c in mode):
        raise NotImplementedError('Piecewise Chebfun3 factors are not supported.')
    return Chebfun3(cols=[c.funs[0].tech for c in factors[0]],
                    rows=[c.funs[0].tech for c in factors[1]],
                    tubes=[c.funs[0].tech for c in factors[2]],
                    core=f.core, domain=d)
