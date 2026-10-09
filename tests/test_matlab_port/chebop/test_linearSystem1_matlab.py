"""Literal21 slots (30 scalar comparisons) of the coupled linear-system test.

Provenance
----------
MATLAB source: tests/chebop/test_linearSystem1.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Copyright 2017 The University of Oxford and The Chebfun Developers.
All three discretizations, both domains, continuous norms and source bounds.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun, jump
from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


@lru_cache(None)
def _case(discretization, piecewise):
    d = (-float(jnp.pi), float(jnp.pi))
    A = Chebop(lambda x, u, v: [u-v.diff(), u.diff()+v], d)
    A.lbc = lambda u, v: u+1
    A.rbc = lambda u, v: v
    x = chebfun(lambda t: t, domain=d)
    if piecewise:
        A.domain = (d[0], 0.0, d[1])
    u = A.solve(0, discretization=discretization)
    u1, u2 = u[0], u[1]
    return u1, u2, x, A.lbc(u1, u2), A.rbc(u1, u2), d


@pytest.mark.parametrize('clause', range(1, 22),
                         ids=[f'source_clause_{i:02d}' for i in range(1, 22)])
def test_literal_linear_system1(clause):
    if clause <= 9:
        group, predicate = divmod(clause-1, 3)
        piecewise = False
    else:
        group, predicate = divmod(clause-10, 4)
        piecewise = True
    discretization = ('chebcolloc1', 'chebcolloc2', 'ultraS')[group]
    u1, u2, x, bc_left, bc_right, d = _case(discretization, piecewise)
    if predicate == 0:
        assert (u1-x.cos()).norm(jnp.inf) < (2000 if piecewise else 100)*TOL
    elif predicate == 1:
        assert (u2-x.sin()).norm(jnp.inf) < (2000 if piecewise else 100)*TOL
    elif predicate == 2:
        assert jnp.abs(bc_left(d[0])) < TOL
        assert jnp.abs(bc_right(d[-1])) < TOL
    else:
        assert jnp.abs(jump(u1, 0)) < TOL
        assert jnp.abs(jump(u2, 0)) < TOL
