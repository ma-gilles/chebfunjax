"""Orientation-aware MATLAB multiplication, exposed through Python ``@``.

Provenance
----------
MATLAB source : @chebfun/mtimes.m, @chebfun2/outerProduct.m,
    @separableApprox/mtimes.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp


def _columns(f):
    from .linalg import Quasimatrix
    cols = f.cols if isinstance(f, Quasimatrix) else (
        [f] if f.n_columns == 1 else f.mat2cell())
    return [c.T if c.is_transposed else c for c in cols]


def _assemble(cols, row=False):
    from .linalg import Quasimatrix
    if row:
        cols = [c.T for c in cols]
    return Quasimatrix(cols, cols[0].domain)


def _outer(f, g):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    fc, gc = _columns(f), _columns(g)
    if len(fc) != len(gc):
        raise ValueError('CHEBFUN:CHEBFUN2:outerProduct:sizes: '
                         'Sizes not consistent for outer product.')
    if any(len(c.funs) != 1 for c in fc + gc):
        # The current separable storage contains single-interval techs.
        raise NotImplementedError('Piecewise Chebfun2 factors are not supported.')
    return Chebfun2.from_cdr(fc, jnp.ones(len(fc)), gc)


def mtimes(f, g):
    """Compute MATLAB ``f*g`` with explicit row/column orientation.

    Python vectors on the right denote column vectors, except for a scalar
    column Chebfun, where they denote a row of multiplicative constants.
    Use two-dimensional arrays to specify matrix shapes without ambiguity.

    Provenance
    ----------
    MATLAB source : @chebfun/mtimes.m, @chebfun2/outerProduct.m,
        @separableApprox/mtimes.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.autodiff.adchebfun import ADChebfun
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2

    from .chebfun import Chebfun, _is_empty_operand
    from .linalg import Quasimatrix

    # Native ADChebfun declares Chebfun inferior, so AD owns either ordering.
    if isinstance(f, Chebfun) and isinstance(g, ADChebfun):
        return g.mtimes(f)
    types = (Chebfun, Quasimatrix)
    ff, gf = isinstance(f, types), isinstance(g, types)
    if ((isinstance(f, Chebfun) and f.isempty()) or
            (isinstance(g, Chebfun) and g.isempty()) or
            (not ff and not isinstance(f, Chebfun2) and _is_empty_operand(f)) or
            (not gf and not isinstance(g, Chebfun2) and _is_empty_operand(g))):
        return Chebfun.empty()
    if isinstance(f, Chebfun2) and gf:
        if g.is_transposed:
            raise ValueError('CHEBFUN:CHEBFUN:mtimes:dims: Matrix dimensions must agree.')
        c, d, r = f.cdr()
        gc = _columns(g)
        weights = jnp.stack([jnp.stack([a.inner(b) for b in gc]) for a in r])
        return mtimes(_assemble(c), jnp.asarray(d) @ weights)
    if ff and isinstance(g, Chebfun2):
        return mtimes(g.T, f.T).T
    if ff and gf:
        if f.is_transposed == g.is_transposed:
            return f * g
        if not f.is_transposed:
            return _outer(f, g)
        fc, gc = _columns(f), _columns(g)
        return jnp.stack([jnp.stack([a.conj().inner(b) for b in gc]) for a in fc])
    if not ff and gf:
        try:
            a = jnp.asarray(f)
        except (TypeError, ValueError):
            raise TypeError('CHEBFUN:CHEBFUN:mtimes:unknown: '
                            'Undefined function mtimes for these input types.') from None
        if a.dtype.kind not in 'biufc':
            raise TypeError('CHEBFUN:CHEBFUN:mtimes:unknown: '
                            'Undefined function mtimes for these input types.')
        if a.ndim == 1:
            a = a[:, None] if len(_columns(g)) == 1 else a[None, :]
        return mtimes(g.T, a.T).T
    if ff:
        try:
            a = jnp.asarray(g)
        except (TypeError, ValueError):
            raise TypeError('CHEBFUN:CHEBFUN:mtimes:unknown: '
                            'Undefined function mtimes for these input types.') from None
        if a.dtype.kind not in 'biufc':
            raise TypeError('CHEBFUN:CHEBFUN:mtimes:unknown: '
                            'Undefined function mtimes for these input types.')
        if a.size == 1:
            return f * a.reshape(())
        n = len(_columns(f))
        if a.ndim == 1:
            a = a[None, :] if n == 1 else a[:, None]
        if a.ndim != 2 or f.is_transposed or a.shape[0] != n:
            raise ValueError('CHEBFUN:CHEBFUN:mtimes:dims: Matrix dimensions must agree.')
        if isinstance(f, Chebfun):
            funs = [p._apply_unary(p.tech @ a) for p in f.funs]
            out = Chebfun(funs=funs, domain=f.domain)
            return f._propagate_point_values(
                out, lambda v: jnp.reshape(v, (-1, n)) @ a)
        fc = _columns(f)
        out = []
        for j in range(a.shape[1]):
            col = fc[0] * a[0, j]
            for k in range(1, n):
                col = col + fc[k] * a[k, j]
            out.append(col)
        return _assemble(out)
    return NotImplemented
