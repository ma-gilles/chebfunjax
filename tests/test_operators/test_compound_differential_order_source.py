"""Differential-order and boundary controls for compound expressions.

Provenance
----------
MATLAB source : @linop/deriveContinuity.m, @treeVar/bivariate.m,
                @treeVar/treeVar.m, tests/chebop/test_pcg.m
Chebfun commit: 7574c77

The polynomial solutions and right-hand sides below are derived analytically.
They check boundary/continuity allocation independently of a Krylov solve.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('op, expected', [
    (lambda x, u: -(2 * u.diff()).diff() + u, [2]),
    (lambda x, u: -((2 + x) * u.diff()).diff() + (1 + x**2) * u, [2]),
    (lambda x, u: -(u.diff().diff()).diff() + u.diff(), [3]),
    (lambda x, u: u.diff() + u.diff(2), [2]),
    (lambda x, u, v: [((2 + x) * u.diff() + v).diff(),
                       v.diff() + u], [2, 1]),
])
def test_compound_orders(op, expected):
    n = Chebop(op)
    assert n._piecewise_orders(len(expected)) == expected


@pytest.mark.parametrize('coefficient', ['affine', 'corner'])
def test_compound_piecewise_manufactured_solution(coefficient):
    dom = (-1.0, 0.0, 1.0)
    if coefficient == 'affine':
        a = lambda x: 2 + x  # noqa: E731
        c = lambda x: 1 + x**2  # noqa: E731
        rhs = lambda x: 5 + 4 * x - x**4  # noqa: E731
    else:
        a = lambda x: 2 + abs(x)  # noqa: E731
        c = lambda x: 1 + 0 * x  # noqa: E731
        rhs = lambda x: 5 + 4 * jnp.abs(x) - x**2  # noqa: E731
    n = Chebop(lambda x, u: -(a(x) * u.diff()).diff() + c(x) * u,
               domain=dom)
    n.bc = 0.0
    f = cj.chebfun(rhs, domain=dom)
    exact = cj.chebfun(lambda x: 1 - x**2, domain=dom)
    u = n.solve(f, n=32)
    assert tuple(u.domain.breakpoints) == dom
    tol = 100 * ChebopPref().bvpTol
    assert float((u - exact).norm(2)) < tol
    assert abs(float(u(-1.0))) < tol
    assert abs(float(u(1.0))) < tol
    assert float((n.op(cj.chebfun('x', domain=dom), u) - f).norm(2)) < tol
