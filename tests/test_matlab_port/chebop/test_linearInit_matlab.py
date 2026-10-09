"""All6 original linear-init clauses, including the native variable quirks.

Provenance
----------
MATLAB source: tests/chebop/test_linearInit.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright 2017 The University of Oxford and The Chebfun Developers.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop

TOL = 1e3*5e-13


def _scalar_operator():
    domain = (0., float(jnp.pi))
    operator = Chebop(lambda x, u: u.diff(2)+x*u, domain)
    operator.lbc, operator.rbc = 2., 3.
    x = chebfun(lambda t: t, domain=domain)
    operator.init = (20*x).sin()
    return operator, x.sin()


@lru_cache(None)
def _scalar(piecewise):
    operator, rhs = _scalar_operator()
    if piecewise:
        operator.domain = (0., 1., float(jnp.pi))
    solution = operator.solve(rhs, discretization='ultraS' if piecewise else 'chebcolloc2')
    return operator, rhs, solution


@lru_cache(None)
def _coupled():
    N, _ = _scalar_operator()
    N.domain = (0., 1., float(jnp.pi))
    domain = (-float(jnp.pi), float(jnp.pi))
    A = Chebop(lambda x, u, v: [u-v.diff(), u.diff()+v], domain)
    A.lbc, A.rbc = lambda u, v: u+1, lambda u, v: v
    x = chebfun('x', domain=domain)
    # Source assigns N.init, not A.init. Retain that literal quirk.
    N.init = [(20*x).cos(), (20*x).sin()]
    rhs = [(-x**2).exp(), 1+(-x**2).exp()]
    solution = A.solve(rhs, discretization='ultraS')
    return A, rhs, solution, domain


@pytest.mark.parametrize('clause', range(1, 7), ids=[f'source_clause_{i:02d}' for i in range(1, 7)])
def test_literal_linear_init(clause):
    if clause <= 4:
        N, rhs, u = _scalar(clause >= 3)
        if clause in (1, 3):
            error = (N(u)-rhs).norm()
        else:
            # Native err4 tests u1 again, even after solving for u2.
            _, _, u1 = _scalar(False)
            error = jnp.abs(u1(0.)-2)+jnp.abs(u1(float(jnp.pi))-3)
    else:
        A, rhs, uv, domain = _coupled()
        if clause == 5:
            error = ChebMatrix([[r-f] for r, f in zip(A(uv), rhs)]).norm()
        else:
            error = jnp.abs(A.lbc(*uv)(domain[0]))+jnp.abs(A.rbc(*uv)(domain[-1]))
    assert error < TOL


def test_independent_u2_boundary_residual():
    _, _, u2 = _scalar(True)
    assert jnp.abs(u2(0.)-2)+jnp.abs(u2(float(jnp.pi))-3) < TOL
