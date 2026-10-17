"""JAX interpolation paths of Chebfun7574c77 @chebfun/interp1.m.

The public adapter is eager: its nodes, domain and interpolation method are
input metadata. Original source: Copyright 2017 by The University of Oxford
and The Chebfun Developers. Polynomial and trigonometric barycentric evaluation and
piecewise-linear sample arithmetic remain JAX operations.
"""
import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.utils.interpolation import bary, bary_weights
from chebfunjax.utils.quadrature import trigpts
from chebfunjax.utils.trigutils import trigBary


def interp1(x, y, method=None, domain=None):
    """Native input normalization and method dispatch; vectors use 1-D adapter.

    Provenance
    ----------
    MATLAB source: @chebfun/interp1.m
    Chebfun commit: 7574c77
    """
    from .chebfun import Chebfun, FevalError, chebfun

    if method is not None and not isinstance(method, str):
        if domain is None:
            domain = method
        elif jnp.size(method) != 0:
            raise FevalError('CHEBFUN:CHEBFUN:interp1:method', 'Unknown interpolation method.')
        method = None
    if method is None or method == '':
        method = 'poly'
    # Preserve the established Python case-insensitive spelling adapter.
    method = method.lower()
    nodes = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    order = jnp.argsort(nodes, stable=True)
    nodes = nodes[order]
    if isinstance(y, Chebfun):
        values = y(nodes)
    else:
        values = jnp.asarray(y)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values.T
        values = values[order]
    # Source one-column sample vectors define a scalar interpolant. The
    # Python vector adapter must not retain an extra array-valued axis when
    # an explicit row vector has just been normalized to a column.
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if domain is None or (not isinstance(domain, Domain) and jnp.size(domain) == 0):
        domain = (float(nodes[0]), float(nodes[-1]))
    requested = domain if isinstance(domain, Domain) else Domain(tuple(float(v) for v in domain))
    if method == 'poly':
        weights = bary_weights(nodes)
        return chebfun(lambda z: bary(jnp.atleast_1d(z), values, nodes, weights),
                       domain=requested.breakpoints, n=len(nodes))
    if method in ('trig', 'periodic'):
        ends = jnp.asarray([nodes[0], nodes[-1]])
        dom = jnp.asarray(requested.breakpoints)
        if bool(jnp.linalg.norm(ends-dom)/(dom[-1]-dom[0]) < 100*jnp.finfo(jnp.float64).eps):
            values = values.at[0].set((values[0]+values[-1])/2)
            nodes, values = nodes[:-1], values[:-1]
        targets = trigpts(len(nodes), (requested.a, requested.b))[0]
        sampled = trigBary(targets, values, nodes, (requested.a, requested.b))
        return chebfun(sampled, domain=requested.breakpoints, trig=True)
    if method == 'linear':
        return linear(nodes, values, requested)
    if method == 'spline':
        return Chebfun.spline(nodes, values, requested.breakpoints)
    if method in ('pchip', 'cubic'):
        return Chebfun.pchip(nodes, values, requested.breakpoints)
    raise FevalError('CHEBFUN:CHEBFUN:interp1:method', f"Unknown method '{method}'.")


def linear(nodes, values, domain):
    """Native union grid, two samples per interval and optional restriction.

    Provenance
    ----------
    MATLAB source: @chebfun/interp1.m, interp1Linear
    Chebfun commit: 7574c77
    """
    from .chebfun import Chebfun

    requested = domain if isinstance(domain, Domain) else Domain(tuple(float(v) for v in domain))
    nodes, values = jnp.asarray(nodes), jnp.asarray(values)
    breaks = jnp.unique(jnp.concatenate([jnp.asarray(requested.breakpoints), nodes]))
    columns = values[:, None] if values.ndim == 1 else values

    def evaluate(z):
        query = jnp.atleast_1d(z)
        out = jax.vmap(lambda column: jnp.interp(query, nodes, column,
                                                left=jnp.nan, right=jnp.nan),
                       in_axes=1, out_axes=-1)(columns)
        return out[:, 0] if values.ndim == 1 else out

    result = Chebfun.from_function(evaluate, Domain(tuple(float(v) for v in breaks)), n=2)
    if requested.a > float(nodes[0]) or requested.b < float(nodes[-1]):
        result = result.restrict(requested.breakpoints)
    return result
