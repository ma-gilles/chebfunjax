"""Literal linear slots1–8 of test_bc for each native backend preference.

Provenance
----------
MATLAB source: tests/chebop/test_bc.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright 2017 The University of Oxford and The Chebfun Developers.
Nonlinear slots9–10 belong to the separate general-BC package.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@lru_cache(None)
def _case(number, backend):
    if number == 0:
        domain = (-3., 4.)
        A = Chebop(lambda x, u: u.diff(2)+4*u.diff()+u, domain)
        A.lbc, A.rbc = -1., 'neumann'
        f = chebfun('exp(sin(x))', domain=domain)
        u = A.solve(f, discretization=backend)
        return ((u.diff(2)+4*u.diff()+u-f).norm(),
                jnp.abs(u(-3.)+1), jnp.abs(u.diff()(4.)))
    if number == 1:
        domain = (-1., 0.)
        A = Chebop(lambda x, u: u.diff(2)+4*u.diff()+200*u, domain)
        A.lbc, A.rbc = lambda u: [u.diff()+2*u-1], lambda u: u.diff()
        f = chebfun('x.*sin(3*x).^2', domain=domain)
        u = A.solve(f, discretization=backend)
        du = u.diff()
        return ((u.diff(2)+4*du+200*u-f).norm(),
                jnp.abs(du(-1.)+2*u(-1.)-1), jnp.abs(du(0.)))
    domain = (-2., 1.)
    N = Chebop(lambda x, u: u.diff(2)+u, domain)
    N.bc = lambda x, u: [u.diff()(0.), u.sum()]
    x = chebfun(lambda t: t, domain=domain)
    rhs = x.sin()
    u = N.solve(rhs, discretization=backend)
    return ((N(u)-rhs).norm(), jnp.linalg.norm(jnp.asarray(N.bc(x, u))))


@pytest.mark.parametrize('backend', ['chebcolloc2', 'chebcolloc1', 'ultraS'])
@pytest.mark.parametrize('clause', range(1, 9), ids=[f'source_clause_{i:02d}' for i in range(1, 9)])
def test_literal_linear_boundary_conditions(clause, backend):
    group, offset = (divmod(clause-1, 3) if clause <= 6 else (2, clause-7))
    assert _case(group, backend)[offset] < 1e-10
