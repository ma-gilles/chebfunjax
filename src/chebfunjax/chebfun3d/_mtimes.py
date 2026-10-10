"""Native mode-one contractions and scalar matrix products.

Provenance
----------
MATLAB source : @chebfun3/{mtimes,tucker,txm}.m, @chebfun/innerProduct.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.mtimes import _columns
from chebfunjax.domain import Domain


def _panel(factors, interval):
    domain = Domain(interval)
    return Chebfun.horzcat(*[Chebfun([_Piece(tech, interval)], domain) for tech in factors])


def _inner(left, right):
    return jnp.stack([jnp.stack([f.inner(g) for g in _columns(right)]) for f in _columns(left)])


def _native_squeeze(tensor):
    # MATLAB drops trailing singleton dimensions but always has at least two.
    shape = list(tensor.shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    tensor = tensor.reshape(tuple(shape))
    if len(shape) <= 2:
        return tensor
    tensor = jnp.squeeze(tensor)
    if tensor.ndim < 2:
        tensor = tensor.reshape((-1, 1))
    return tensor


def source_mtimes(f, g):
    """MATLAB ``f*g`` exposed as ``f.mtimes(g)`` and Python ``f @ g``."""
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
    from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

    if isinstance(g, Chebfun3):
        raise ValueError('CHEBFUN:CHEBFUN3:mtimes: CHEBFUN3 does not support this operation.')
    if isinstance(g, Chebfun3v):
        return Chebfun3v(components=[f*c for c in g.components])
    if isinstance(g, (Chebfun, Chebfun2)):
        left = _panel(f.cols, tuple(f.domain[:2]))
        if isinstance(g, Chebfun):
            x = _inner(left, g)
            temp = _native_squeeze(jnp.tensordot(x.T, f.core, axes=(1, 0)))
            if temp.ndim != 2:
                raise ValueError('Native mtimes SVD requires a matrix after squeeze.')
            u, s, vh = jnp.linalg.svd(temp, full_matrices=True)
            rows = _panel(f.rows, tuple(f.domain[2:4])) @ u
            tubes = _panel(f.tubes, tuple(f.domain[4:6])) @ jnp.conj(vh).T
            return Chebfun2.from_pivot_values(_columns(tubes), 1/s, _columns(rows), f.domain[2:6])
        gr = _panel(g.approx.rows, tuple(g.domain[:2]))
        x = _inner(left, gr)
        temp = _native_squeeze(jnp.tensordot(x.T, f.core, axes=(1, 0)))
        # Native gPivots is diag(1./g.pivotValues), without CDR clipping.
        pivots = jnp.diag(1/g.pivot_values)
        if pivots.shape[1] != temp.shape[0]:
            raise ValueError("CHEBFUN:CHEBFUN3:txm: Tensor-matrix dimensions do not agree")
        core = jnp.tensordot(pivots, temp, axes=(1, 0))
        if core.ndim == 2:
            core = core[:, :, None]
        return Chebfun3(cols=list(g.approx.cols), rows=list(f.rows), tubes=list(f.tubes),
                        core=core, domain=tuple(g.domain[2:4])+tuple(f.domain[2:6]))
    try:
        scalar = jnp.asarray(g)
    except (TypeError, ValueError):
        raise TypeError('CHEBFUN:CHEBFUN3:mtimes:unknown: Undefined function mtimes for these input types.') from None
    if scalar.dtype.kind not in 'biufc':
        raise TypeError('CHEBFUN:CHEBFUN3:mtimes:unknown: Undefined function mtimes for these input types.')
    if scalar.size != 1:
        raise ValueError('CHEBFUN:CHEBFUN3:mtimes:size: Sizes are inconsistent.')
    scalar = scalar.reshape(())
    if bool(scalar == 0):
        return chebfun3(lambda x, y, z: 0*x, f.domain)
    return Chebfun3(cols=list(f.cols), rows=list(f.rows), tubes=list(f.tubes),
                    core=f.core*scalar, domain=f.domain)
