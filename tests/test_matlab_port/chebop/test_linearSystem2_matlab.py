"""All12 literal slots /19 comparisons of the second coupled system test.

Provenance
----------
MATLAB source: tests/chebop/test_linearSystem2.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Copyright2017 The University of Oxford and The Chebfun Developers.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun, jump
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop

TOL = 1e-10
D = (-1., 1.)


def _myop(x, u):
    return [u[0].diff()+u[0]+2*u[1], u[0].diff()-u[0]+u[1].diff()]


@lru_cache(None)
def _case(group):
    if group == 4:
        A = Chebop(lambda u: [u[0].diff()+u[0]+2*u[1],
                              u[0].diff()-u[0]+u[1].diff()], D)
    elif group >= 2:
        A = Chebop(_myop, D)
    else:
        A = Chebop(lambda x, u: [u[0].diff()+u[0]+2*u[1],
                                 u[0].diff()-u[0]+u[1].diff()], D)
    A.lbc = lambda u: u[0]+u[0].diff()
    A.rbc = lambda u: u[1].diff()
    if group in (2, 3):
        A.numVars = 2
    if group in (1, 3):
        A.domain = (-1., 0., 1.)
    x = chebfun(lambda t: t, domain=D)
    f = [x.exp(), chebfun(1., domain=D)]
    u = A.solve(f, discretization='ultraS' if group >= 2 else None)
    residual = ChebMatrix([[r-g] for r, g in zip(A(u), f)]).norm()
    return u, residual, A.lbc(*u)(D[0]), A.rbc(*u)(D[-1])


@pytest.mark.parametrize('clause', range(1, 13),
                         ids=[f'source_clause_{i:02d}' for i in range(1, 13)])
def test_literal_linear_system2(clause):
    groups = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1), 5: (1, 2),
              6: (2, 0), 7: (2, 1), 8: (3, 0), 9: (3, 1), 10: (3, 2),
              11: (4, 0), 12: (4, 1)}
    group, predicate = groups[clause]
    u, residual, left, right = _case(group)
    if predicate == 0:
        assert residual < TOL
    elif predicate == 1:
        assert jnp.abs(left) < TOL
        assert jnp.abs(right) < TOL
    else:
        assert jnp.abs(jump(u[0], 0.)) < TOL
        assert jnp.abs(jump(u[1], 0.)) < TOL
