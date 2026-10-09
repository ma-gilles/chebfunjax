"""Typed composition dispatch and source composition of two Chebfuns.

Provenance
----------
MATLAB source : @chebfun/compose.m, @chebfun/tolUnique.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Adaptive construction and variable breakpoint unions are eager host adapters;
all new numerical operations use JAX. Existing constructor/extrema limitations
remain applicable.
"""
import warnings

import jax.numpy as jnp


def periodic(f):
    """Source isPeriodicTech tests the first FUN; quasimatrices test columns."""
    from chebfunjax.tech.trigtech import Trigtech

    from .linalg import Quasimatrix
    if isinstance(f, Quasimatrix):
        return all(periodic(c) for c in f.cols)
    return isinstance(getattr(f.funs[0], 'tech', None), Trigtech)


def _columns(f):
    from .linalg import Quasimatrix
    if isinstance(f, Quasimatrix):
        return f.cols
    if f.n_columns == 1:
        return [f]
    # Source extractColumns slices stored pointValues as well as each FUN.
    return [c.set_point_values(f.point_values[:, k])
            for k, c in enumerate(f.mat2cell())]


def _assemble(columns):
    from .linalg import Quasimatrix
    return columns[0] if len(columns) == 1 else Quasimatrix(columns, columns[0].domain)


def _two(f, outer, pref):
    """Source composeTwoChebfuns(f, outer): outer(f), including pointValues."""
    from .chebfun import Chebfun, chebfun
    from .inverse import tol_union
    if f.isempty() or outer.isempty():
        return chebfun()
    if f.is_transposed != outer.is_transposed:
        raise ValueError('CHEBFUN:CHEBFUN:compose:composeTwoChebfuns:trans: '
                         'Cannot compose a row CHEBFUN with a column CHEBFUN.')
    transposed = f.is_transposed
    if transposed:
        f, outer = f.T, outer.T
    if not f.isreal():
        warnings.warn('CHEBFUN:CHEBFUN:compose:composeTwoChebfuns:complex: '
                      'F should be real valued to construct G(F). Results may '
                      'be inaccurate if G is not a polynomial.', stacklevel=3)
    else:
        tol = 1000*jnp.finfo(jnp.float64).eps*jnp.maximum(jnp.max(f.vscale), jnp.max(outer.vscale))
        hscale = max(abs(f.domain.a), abs(f.domain.b))
        if hscale == float('inf'):
            hscale = 1.
        low, high = f.minandmax()
        minimum, maximum = jnp.min(jnp.asarray(low[1])), jnp.max(jnp.asarray(high[1]))
        if bool(outer.domain.a > minimum + tol*hscale) or bool(outer.domain.b < maximum-tol*hscale):
            raise ValueError('CHEBFUN:CHEBFUN:compose:domain: Range of F must be in the domain of G.')
    if f.deltas or outer.deltas:
        warnings.warn('CHEBFUN:CHEBFUN:compose:composeTwoChebfuns:deltas: '
                      'Composition ignores delta functions. Results may not make any sense.', stacklevel=3)
    if len(outer.domain.breakpoints) > 2:
        breaks = [jnp.asarray(f.domain.breakpoints)]
        for value in outer.domain.breakpoints[1:-1]:
            breaks.append(jnp.ravel((f-value).roots()))
        # tolUnique and tolUnion(A,[]) have the same default threshold and
        # simultaneous close-pair averaging/deletion, including source quirks.
        domain = tol_union(jnp.concatenate(breaks), jnp.asarray([]))
        f = f.restrict(domain)
    result = f.compose(lambda value: outer(value), pref=pref)
    result = result.set_point_values(outer(f(jnp.asarray(result.domain.breakpoints))))
    return Chebfun._as_transposed(result, transposed)


def compose_object(f, op, pref=None):
    """Return (handled, result) for source Chebfun/2D/3D typed operators."""
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

    from .chebfun import Chebfun, chebfun
    from .linalg import Quasimatrix
    if isinstance(op, SeparableApprox):
        op = Chebfun2(approx=op)
    if not isinstance(op, (Chebfun, Quasimatrix, Chebfun2, Chebfun2v, Chebfun3, Chebfun3v)):
        return False, None
    if isinstance(f, Chebfun) and f.isempty():
        return True, f
    if isinstance(op, (Chebfun, Quasimatrix)):
        nf = f.n_cols if isinstance(f, Quasimatrix) else f.n_columns
        ng = op.n_cols if isinstance(op, Quasimatrix) else op.n_columns
        if nf > 1 and ng > 1:
            raise ValueError('CHEBFUN:CHEBFUN:compose:trans: Cannot compose two array-valued CHEBFUN objects.')
        if isinstance(f, Quasimatrix) or isinstance(op, Quasimatrix):
            fc, gc = _columns(f), _columns(op)
            results = [_two(c, gc[0], pref) for c in fc] if len(fc)>1 else [_two(fc[0], c, pref) for c in gc]
            return True, _assemble(results)
        return True, _two(f, op, pref)
    dim = 2 if isinstance(op, (Chebfun2, Chebfun2v)) else 3
    vector = isinstance(op, (Chebfun2v, Chebfun3v))
    # A row Chebfun has infinite second dimension in source size(f,2).
    if f.is_transposed:
        label = f'Cheb{dim}'+('V' if vector else '')+'ofCheb'
        raise ValueError('CHEBFUN:CHEBFUN:compose:'+label+': Wrong number of inner columns.')
    columns = _columns(f)
    if dim == 2 and len(columns) == 1:
        columns = [columns[0].real(), columns[0].imag()]
    if len(columns) != dim:
        label = f'Cheb{dim}'+('V' if vector else '')+'ofCheb'
        raise ValueError('CHEBFUN:CHEBFUN:compose:'+label+': Wrong number of inner columns.')
    if not all(c.isreal() for c in columns):
        raise ValueError(f'CHEBFUN:CHEBFUN:compose:complex{dim}: Inner columns must be real.')
    def one(outer):
        return chebfun(lambda t: outer(*(c(t) for c in columns)),
                       domain=columns[0].domain.breakpoints, trig=all(periodic(c) for c in columns))
    if vector:
        components = op.components
        return True, _assemble([one(c) for c in components])
    return True, one(op)
