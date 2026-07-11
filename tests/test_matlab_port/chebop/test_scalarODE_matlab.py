"""Port of MATLAB Chebfun tests/chebop/test_scalarODE.m (Fable 5).

Nonlinear scalar BVP u'' + sin(u - 0.2) = 0, u(0)=2, u(pi)=3.  MATLAB
solves it under three discretizations; chebfunjax has one -- the
residual + BC assertions are the same.

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop

TOL = 1e3 * 1e-10  # 1e3 * bvpTol


class TestChebopScalarODE:
    def test_nonlinear_bvp_residual_and_bcs(self):
        N = Chebop(lambda x, u: u.diff(2) + (u - 0.2).sin(),
                   domain=(0.0, float(np.pi)))
        N.lbc = 2.0
        N.rbc = 3.0
        u = N.solve(0.0)
        assert abs(float(u(jnp.asarray(0.0))) - 2.0) < TOL
        assert abs(float(u(jnp.asarray(float(np.pi)))) - 3.0) < TOL
        # residual: u'' + sin(u - .2) ~ 0 at interior points
        xs = jnp.asarray(np.linspace(0.2, np.pi - 0.2, 30))
        res = u.diff(2)(xs) + jnp.sin(u(xs) - 0.2)
        assert float(jnp.max(jnp.abs(res))) < 1e3 * TOL

    def test_other_discretizations(self):
        pytest.skip("chebfunjax has a single collocation discretization")
