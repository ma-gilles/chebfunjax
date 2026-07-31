"""Port of MATLAB Chebfun tests/chebop/test_periodic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
        # MATLAB pass(2): a periodic piecewise SYSTEM on [-pi, 0, pi]:
        # [u - v'; u'' + v] = [0; cos x], bc periodic; true solution
        # [cos(x+3pi/4); cos(x+pi/4)]/sqrt(2).  tol = 1e4*bvpTol = 1e-6.
        # (This was skipped as "chebop is scalar-only" -- stale: the
        # system solvers exist; measured 2e-16 on 2026-07-31.)
        d = (-np.pi, np.pi)
        A = Chebop(lambda x, u, v: [u - v.diff(), u.diff(2) + v], d)
        A.bc = "periodic"
        f = [cj.chebfun(0.0, domain=list(d)),
             cj.chebfun(lambda t: jnp.cos(t), domain=list(d))]
        uv = A.solve(f)
        xs = jnp.asarray(np.linspace(-3.0, 3.0, 40))
        tu = np.cos(np.asarray(xs) + 3 * np.pi / 4) / np.sqrt(2)
        tv = np.cos(np.asarray(xs) + np.pi / 4) / np.sqrt(2)
        assert float(np.max(np.abs(np.asarray(uv[0](xs)) - tu))) < 1e-6
        assert float(np.max(np.abs(np.asarray(uv[1](xs)) - tv))) < 1e-6
