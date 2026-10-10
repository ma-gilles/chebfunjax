"""Port of MATLAB Chebfun tests/chebop/test_chap21.m (Fable 5).

This module covers all four asserted source predicates: differentiation
matrix identities and size (pass(1) and pass(2)), operator application
(pass(3)), and the adaptive Dirichlet solve residual (pass(4)). The remaining
constructions, oldschool finite-grid solves, and plots in the MATLAB file are
demonstrations without assertions and are not represented as tested here.

Provenance
----------
MATLAB source : tests/chebop/test_chap21.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


class TestChebopChap21:
    def test_operator_application(self):
        # pass(3): L = diff(u,2) + diff(u) + 100 u applied to exp(x) is 102 exp(x).
        # FIXED (Fable 5): scalar operator application L(f) (MATLAB L*f).
        L = Chebop(lambda u: u.diff(2) + u.diff() + 100 * u, (-1.0, 1.0))
        f = cj.chebfun(lambda x: jnp.exp(x))
        Lf = L(f)
        exact = cj.chebfun(lambda x: 102 * jnp.exp(x))
        assert float((Lf - exact).norm()) < 1e-10

    def test_diffmat_realization(self):
        # pass(1,2): D(14), size(D(33)) -- finite differentiation matrices.
        # FIXED (Fable 5): Chebop.matrix(n) / N(n) realize the dense n x n
        # collocation (differentiation) matrix; D(14) @ p(x14) reproduces p'.
        import numpy as np

        from chebfunjax.utils.quadrature import chebpts
        D = Chebop(lambda u: u.diff(), (-1.0, 1.0))
        # D(14) via integer-call syntax and via matrix() must agree.
        D14 = np.asarray(D(14))
        assert np.allclose(D14, np.asarray(D.matrix(14)))
        x14 = np.asarray(chebpts(14))
        p = cj.chebfun(lambda x: jnp.sin(x), domain=(-1.0, 1.0))
        pv = np.asarray(p(jnp.asarray(x14)))
        pp14 = np.asarray(p.diff()(jnp.asarray(x14)))
        assert np.linalg.norm(pp14 - D14 @ pv) < 1e-10
        # size(D(33)) == 33 x 33
        assert np.asarray(D(33)).shape == (33, 33)

    def test_stiff_solve_residual(self):
        # pass(4): native adaptive Dirichlet solve, with default solver settings.
        L = Chebop(lambda u: u.diff(2) + u.diff() + 100 * u, (-1.0, 1.0))
        L.bc = "dirichlet"
        x = cj.chebfun(lambda x: x)
        u = L.solve(x)
        assert float((L(u) - x).norm()) < 1e-10
