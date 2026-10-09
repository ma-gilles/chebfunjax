"""Literal14 slots /40 comparisons of native scalar linear ODE tests.

Provenance
----------
MATLAB source: tests/chebop/test_linearScalarODEs.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright 2017 The University of Oxford and The Chebfun Developers.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun, jump
from chebfunjax.operators.chebop import Chebop

BVP_TOL = 5e-13
BACKENDS = ('chebcolloc2', 'ultraS', 'chebcolloc1')


@lru_cache(None)
def _case(number):
    domain = (0., float(jnp.pi)) if number == 0 else (-1., 0., float(jnp.pi))
    if number == 0:
        operator = Chebop(lambda x, u: u.diff(2)+x*u, domain)
    elif number == 1:
        operator = Chebop(lambda x, u: u.diff(2)+x.cos()*u, domain)
    else:
        operator = Chebop(lambda x, u: u.diff(2)+abs(x-1)*u, domain)
    operator.lbc = 2.
    operator.rbc = 3. if number == 0 else -1.
    x = chebfun(lambda t: t, domain=domain)
    rhs = x.sin()
    solutions = [operator.solve(rhs, discretization=backend) for backend in BACKENDS]
    return operator, rhs, solutions, domain


@pytest.mark.parametrize('clause', range(1, 15), ids=[f'source_clause_{i:02d}' for i in range(1, 15)])
def test_literal_linear_scalar_odes(clause):
    if clause <= 4:
        case = 0
    elif clause <= 9 or clause == 14:
        case = 1
    else:
        case = 2
    operator, rhs, solutions, domain = _case(case)
    tol = (1e1 if case == 2 else 1e3)*BVP_TOL
    starts = {0: 1, 1: 5, 2: 10}
    if starts[case] <= clause < starts[case]+3:
        index = clause-starts[case]
        u = solutions[index]
        factor = (50 if index == 0 else 10) if case == 2 else 1
        assert (operator(u)-rhs).norm() < factor*tol
        # Native tests intentionally use signed endpoint inequalities.
        assert u(domain[0])-2 < tol
        assert u(domain[-1])-(3 if case == 0 else -1) < tol
    elif clause in (8, 13):
        for u in solutions:
            assert jnp.abs(jump(u, 0.)) < tol
    elif clause in (4, 9):
        for i, j in ((0, 1), (1, 2), (0, 2)):
            assert (solutions[i]-solutions[j]).norm() != 0
    else:
        # Literal slot14 repeats u5/u6 from the preceding problem.
        assert (solutions[1]-solutions[2]).norm() != 0


@pytest.mark.parametrize('case', range(3))
def test_independent_absolute_endpoint_residuals(case):
    _operator, _rhs, solutions, domain = _case(case)
    tol = (1e1 if case == 2 else 1e3)*BVP_TOL
    for u in solutions:
        assert jnp.abs(u(domain[0])-2) < tol
        assert jnp.abs(u(domain[-1])-(3 if case == 0 else -1)) < tol
