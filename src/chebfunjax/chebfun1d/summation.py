"""Dimension and variable-limit dispatch for Chebfun integration.

Provenance
----------
MATLAB source : @chebfun/sum.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp


def sum_dispatch(f, args, dim=None):
    """Apply the source sum argument parser and orientation rules.

    Provenance
    ----------
    MATLAB source : @chebfun/sum.m
    Chebfun commit: 7574c77
    """
    from .chebfun import Chebfun
    from .linalg import Quasimatrix

    row = f.is_transposed
    dimension = (2 if row else 1) if dim is None else dim
    limits = None
    if len(args) == 2:
        limits = args
    elif len(args) == 1:
        arg = jnp.asarray(args[0])
        if arg.size == 2:
            limits = tuple(arg.reshape(-1))
        elif arg.size == 1:
            dimension = int(arg.reshape(()))
        else:
            raise ValueError('sum expects a dimension or two limits')
    elif args:
        raise TypeError('sum accepts at most two positional arguments')
    if dimension not in (1, 2):
        raise ValueError('sum dimension must be 1 or 2')
    if row != (dimension == 2):
        if isinstance(f, Quasimatrix):
            out = f.cols[0]
            for c in f.cols[1:]:
                out = out + c
            return out
        if f.isempty() or f.n_columns == 1:
            return f
        pieces = [p._apply_unary(p.tech.sum(2)) for p in f.funs]
        out = Chebfun._as_transposed(Chebfun(funs=pieces, domain=f.domain), row)
        return f._propagate_point_values(out, lambda v: jnp.sum(v, axis=1))
    if isinstance(f, Quasimatrix):
        results = [c.sum(*args, dim=dimension) for c in f.cols]
        if isinstance(results[0], Chebfun):
            return Quasimatrix(results, results[0].domain)
        out = jnp.asarray(results)
        return out.reshape(-1, 1) if row else out
    if f.isempty():
        return jnp.float64(0)
    if limits is None:
        return f.sum()
    a, b = limits
    if ((isinstance(a, Chebfun) and not a.isreal()) or
            (isinstance(b, Chebfun) and not b.isreal()) or
            (not isinstance(a, Chebfun) and jnp.iscomplexobj(a)) or
            (not isinstance(b, Chebfun) and jnp.iscomplexobj(b))):
        raise ValueError('CHEBFUN:CHEBFUN:sum:sumSubDom:complex: '
                         'Chebfun/sum does not support complex limits of integration.')
    if not isinstance(a, Chebfun) and not isinstance(b, Chebfun):
        if min(a, b) < f.domain.a or max(a, b) > f.domain.b:
            raise ValueError('CHEBFUN:CHEBFUN:sum:sumSubDom:ab: Not a valid subdomain.')
        if a == f.domain.a and b == f.domain.b:
            return f.sum()
        primitive = (f.T if row else f).cumsum()
        out = primitive(b) - primitive(a)
        return out.reshape(-1, 1) if row and f.n_columns > 1 else out
    if not isinstance(a, Chebfun) and a < f.domain.a:
        raise ValueError('CHEBFUN:CHEBFUN:sum:sumSubDom:a: Not a valid subdomain.')
    if not isinstance(b, Chebfun) and b > f.domain.b:
        raise ValueError('CHEBFUN:CHEBFUN:sum:sumSubDom:b: Not a valid subdomain.')
    base = f.T if row else f
    if f.n_columns > 1:
        results = [base.extract_columns(k).sum(a, b) for k in range(f.n_columns)]
        out = Quasimatrix(results, results[0].domain)
        return out.T if row else out
    primitive = base.cumsum()
    out = primitive(b) - primitive(a)
    return out.T if row else out
