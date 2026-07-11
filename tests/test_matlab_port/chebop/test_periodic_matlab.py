"""Port of MATLAB Chebfun tests/chebop/test_periodic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e4 * 1e-10


class TestChebopPeriodic:
    def test_periodic_string_bc(self):
        N = Chebop(lambda x, u: u.diff(2) - u)
        N.bc = "periodic"
        rhs = cj.chebfun(lambda x: jnp.sin(np.pi * x))
        v = N.solve(rhs)
        # exact solution: v = -sin(pi x)/(pi^2 + 1)
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 50))
        exact = -jnp.sin(np.pi * xs) / (np.pi ** 2 + 1)
        err = jnp.abs(v(xs) - exact)
        assert float(jnp.max(err)) < TOL
        # periodicity
        assert abs(float(v(jnp.asarray(-1.0)) - v(jnp.asarray(1.0)))) \
            < TOL

    def test_periodic_system(self):
        pytest.skip("chebfunjax chebop is scalar-only")
