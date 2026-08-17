"""Port of MATLAB Chebfun tests/chebop/test_linearScalarODEs.m (Fable 5).

u'' + x u = sin(x) on [0, pi], u(0)=2, u(pi)=3 (one discretization).

Provenance
----------
MATLAB source : tests/chebop/test_linearScalarODEs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e3 * 1e-10


class TestChebopLinearScalarODEs:
    def test_variable_coefficient_bvp(self):
        N = Chebop(lambda x, u: u.diff(2) + x * u,
                   domain=(0.0, float(np.pi)))
        N.lbc = 2.0
        N.rbc = 3.0
        rhs = cj.chebfun(jnp.sin, domain=(0.0, float(np.pi)))
        u = N.solve(rhs)
        assert abs(float(u(jnp.asarray(0.0))) - 2.0) < TOL
        assert abs(float(u(jnp.asarray(float(np.pi)))) - 3.0) < TOL
        xs = jnp.asarray(np.linspace(0.2, np.pi - 0.2, 30))
        res = u.diff(2)(xs) + xs * u(xs) - jnp.sin(xs)
        assert float(jnp.max(jnp.abs(res))) < 1e4 * TOL

    @pytest.mark.parametrize("disc", ["ultraS", "chebcolloc1"])
    def test_other_discretizations(self, disc):
        # MATLAB solves the same BVP under ultraS and chebcolloc1.
        N = Chebop(lambda x, u: u.diff(2) + x * u,
                   domain=(0.0, float(np.pi)))
        N.lbc = 2.0
        N.rbc = 3.0
        rhs = cj.chebfun(jnp.sin, domain=(0.0, float(np.pi)))
        u = N.solve(rhs, n=64, discretization=disc)
        assert abs(float(u(jnp.asarray(0.0))) - 2.0) < TOL
        assert abs(float(u(jnp.asarray(float(np.pi)))) - 3.0) < TOL
        xs = jnp.asarray(np.linspace(0.2, np.pi - 0.2, 30))
        res = u.diff(2)(xs) + xs * u(xs) - jnp.sin(xs)
        assert float(jnp.max(jnp.abs(res))) < 1e4 * TOL
