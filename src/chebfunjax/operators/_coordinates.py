"""Private discrete Chebyshev coordinate capabilities (representation adaptation).

Provenance
----------
MATLAB source: @operatorBlock/operatorBlock.m and @functionalBlock/functionalBlock.m
(algebra), @chebtech/diff.m, @chebtech2/{vals2coeffs,coeffs2vals}.m;
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df. Multiplication interpolates on the ORIGINAL n-point grid,
so composition retains intermediate interpolation/aliasing. This is not
blockCoeff/Leibniz expansion or a new equation discretization.
"""
import math
from numbers import Integral

import jax.numpy as jnp

from chebfunjax.domain import _linear_inverse_map
from chebfunjax.tech.chebtech import _clenshaw, _diff_coeffs
from chebfunjax.utils.quadrature import chebpts


def valid_domain(domain):
    """Bounded affine metadata only; reject unsupported data without changing legacy paths."""
    try:
        return (len(domain) == 2 and all(math.isfinite(float(x)) for x in domain)
                and float(domain[0]) < float(domain[1]))
    except (TypeError, ValueError, OverflowError):
        return False


def eligible(problem):
    """Capabilities own construction domains; nodal assembly owns disc.domain."""
    domain = tuple(problem.domain)
    order = problem.L.order
    return (valid_domain(domain) and isinstance(order, Integral)
            and not isinstance(order, bool) and order >= 0
            and tuple(problem.L.domain) == domain
            and problem.L._coordinate_fn is not None
            and all(tuple(bc.domain) == domain and bc._coordinate_fn is not None
                    for bc in problem.bcs))


def binary(left, right, operation):
    a, b = left._coordinate_fn, right._coordinate_fn
    if a is None or b is None or left.domain != right.domain:
        return None
    if operation == 'add':
        return lambda n: a(n) + b(n)
    if operation == 'sub':
        return lambda n: a(n) - b(n)
    return lambda n: a(n) @ b(n)


def scale(block, scalar):
    fn = block._coordinate_fn
    return None if fn is None else lambda n: scalar * fn(n)


def identity(domain):
    return (lambda n: jnp.eye(n)) if valid_domain(domain) else None


def derivative(domain, order):
    if (not valid_domain(domain) or not isinstance(order, Integral)
            or isinstance(order, bool) or order < 0):
        return None
    def matrix(n):
        c = _diff_coeffs(jnp.eye(n), order) * (2/(domain[1]-domain[0]))**order
        return jnp.pad(c, ((0, n-c.shape[0]), (0, 0)))
    return matrix


def evaluation(x, domain):
    if not valid_domain(domain):
        return None
    return lambda n: _clenshaw(jnp.eye(n), _linear_inverse_map(jnp.asarray(x), *domain))


def multiplier(f, domain):
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    if (not valid_domain(domain) or not isinstance(f, Chebfun)
            or tuple(f.domain.breakpoints) != tuple(domain) or len(f.funs) != 1
            or bool(f.deltas)
            or f.funs[0].coeffs.ndim != 1
            or not isinstance(f.funs[0].tech, (Chebtech1, Chebtech2))
            or not f.funs[0].tech.ishappy
            or not bool(jnp.all(jnp.isfinite(f.funs[0].coeffs)))):
        return None
    point_values = getattr(f, '_point_values', None)
    if point_values is not None:
        point_values = jnp.asarray(point_values)
        # Single scalar piece has exactly two scalar exterior endpoint values.
        if point_values.shape != (2,) or not bool(jnp.all(jnp.isfinite(point_values))):
            return None
    def matrix(n):
        # Local import avoids blocks -> linop -> blocks import cycle.
        from chebfunjax.operators.linop import _source_coeffs2vals, _source_vals2coeffs
        a, b = domain
        t = chebpts(n, kind=2)
        # Match diag's original physical sampling map before evaluating f.
        x = .5*(b-a)*t + .5*(a+b)
        values = _clenshaw(f.funs[0].coeffs, _linear_inverse_map(x, a, b))
        # Source @chebfun/feval.m columnFeval: override only exact physical
        # breakpoint matches, after tech evaluation; jnp.where promotes complex.
        if point_values is not None:
            values = jnp.where(x == a, point_values[0], values)
            values = jnp.where(x == b, point_values[1], values)
        return _source_vals2coeffs(values[:, None] * _source_coeffs2vals(jnp.eye(n)))
    return matrix
