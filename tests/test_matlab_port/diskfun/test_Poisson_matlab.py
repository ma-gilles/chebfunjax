"""Port of MATLAB Chebfun tests/diskfun/test_Poisson.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_Poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-8


class TestDiskfunPoisson:
    def test_manufactured_solution(self):
        # u = (1 - r^2) r^2 cos(2t), u|r=1 = 0;
        # lap u = -12 r^2 cos(2t) (closed form -- NOT the library
        # laplacian, which is broken for angular modes; see the
        # laplacian port).
        u_exact = Diskfun.from_function(
            lambda t, r: (1 - r ** 2) * r ** 2 * jnp.cos(2 * t))
        rhs = Diskfun.from_function(
            lambda t, r: -12 * r ** 2 * jnp.cos(2 * t))
        u = Diskfun.poisson(rhs, m=64)
        t0, r0 = jnp.asarray(0.9), jnp.asarray(0.55)
        assert abs(float(u(t0, r0)) - float(u_exact(t0, r0))) < TOL
