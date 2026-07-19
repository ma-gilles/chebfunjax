"""Core-suite coverage mirrors for :mod:`chebfunjax.operators.chebop`.

Targets the introspection and parameter/interior-BC paths the core suite did
not reach: ``matrix(n)`` realization, ``linop()`` typed block introspection,
a parameter carried in the boundary conditions, and an interior-point
condition.  All problems use small ``n`` and the cheapest linear operators
that exercise the code paths, with closed-form assertions.

Provenance
----------
Mirrors of MATLAB @chebop tests (test_paramODE_inBCs.m,
test_paramODE_linearization.m); Chebfun commit 7574c77.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.blocks import OperatorBlock
from chebfunjax.operators.chebop import Chebop

TOL = 1e-8


def _a(x):
    return jnp.asarray(float(x))


def _deal(sol):
    return sol[0], sol[1]


class TestChebopMatrixLinop:
    def test_matrix_shape(self):
        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0), lbc=0.0, rbc=0.0)
        M = np.asarray(N.matrix(10))
        assert M.shape == (10, 10)
        assert np.all(np.isfinite(M))

    def test_linop_scalar_block_is_operator(self):
        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0), lbc=0.0, rbc=0.0)
        L = N.linop()
        block = L[0]
        assert isinstance(block, OperatorBlock)

    def test_linop_first_order_operator(self):
        # a genuinely differentiated unknown is typed as an OperatorBlock
        N = Chebop(lambda x, u: u.diff() + u, domain=(-1.0, 1.0), lbc=0.0)
        L = N.linop()
        assert isinstance(L[0], OperatorBlock)


class TestChebopParamInBC:
    def test_first_order_param_in_rbc(self):
        # u' = 1 on [-1, 1], u(-1) = 0, u(1) = p  ->  u = 1 + t, p = 2
        dom = (-1.0, 1.0)
        L = Chebop(lambda t, u, p: u.diff() - 1, dom)
        L.lbc = lambda u, p: u
        L.rbc = lambda u, p: u - p
        u, p = _deal(L.solve(0.0, n=24))
        assert isinstance(u, Chebfun) and isinstance(p, Chebfun)
        line = cj.chebfun(lambda t: 1.0 + t, domain=dom)
        npt.assert_allclose(float((u - line).norm()), 0.0, atol=TOL)
        npt.assert_allclose(float(p(_a(0.0))), 2.0, atol=TOL)

    def test_param_determined_implicitly(self):
        # diff(u) - p, u(-1) = 0, u(1) = 2  ->  p = 1, u = 1 + t
        dom = (-1.0, 1.0)
        L = Chebop(lambda x, u, p: u.diff() - p, dom)
        L.lbc = lambda u, p: u
        L.rbc = lambda u, p: u - 2
        u, p = _deal(L.solve(0.0, n=24))
        npt.assert_allclose(float(p(_a(0.0))), 1.0, atol=TOL)
        npt.assert_allclose(float(u(_a(1.0))), 2.0, atol=TOL)


class TestChebopInteriorBC:
    def test_interior_point_condition(self):
        # u'' = -1 on [-1, 1], u(-1) = 0, interior condition u(0) = 1/2
        #   ->  u = 1/2 - x^2/2
        N = Chebop(lambda x, u: u.diff(2), (-1.0, 1.0))
        N.bc = lambda x, u: [u(_a(-1.0)), u(_a(0.0)) - 0.5]
        u = N.solve(-1.0, n=24)
        assert isinstance(u, Chebfun)
        npt.assert_allclose(float(u(_a(-1.0))), 0.0, atol=TOL)
        npt.assert_allclose(float(u(_a(0.0))), 0.5, atol=TOL)
        npt.assert_allclose(float(u(_a(0.5))), 0.5 - 0.5 * 0.25, atol=TOL)
