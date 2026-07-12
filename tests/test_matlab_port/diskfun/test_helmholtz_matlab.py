"""Port of MATLAB Chebfun tests/diskfun/test_helmholtz.m (Fable 5).

FIXED: Diskfun.helmholtz (and bc support in Diskfun.poisson) added in
the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfun/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 2e3 * np.finfo(float).eps
THS = jnp.asarray(np.linspace(-np.pi, np.pi, 13))
RS = jnp.asarray(np.linspace(0.0, 1.0, 9))
TT, RR = jnp.meshgrid(THS, RS, indexing="ij")


def _nrm(u, ex):
    return float(jnp.max(jnp.abs(u(TT, RR) - ex(TT, RR))))


class TestDiskfunHelmholtz:
    def test_k0_matches_poisson(self):
        # pass(1)-(2)
        def tru(t, r):
            return jnp.exp(-r * jnp.cos(t) - r ** 2 * jnp.sin(2 * t))

        truD = Diskfun.from_function(tru)
        rhs = truD.laplacian()

        def bc(t):
            return jnp.exp(-jnp.cos(t) - jnp.sin(2 * t))

        u = Diskfun.poisson(rhs, bc, m=100)
        v = Diskfun.helmholtz(rhs, 0.0, bc, m=100)
        assert _nrm(v, tru) < 2e4 * TOL
        assert _nrm(v, u) < 2e4 * TOL

    def test_various_k(self):
        # pass(3)-(7)
        def utru(t, r):
            x = r * jnp.cos(t)
            y = r * jnp.sin(t)
            return jnp.cos(5 * (x + y) - 0.2) + jnp.sin(3 * x * y)

        uD = Diskfun.from_function(utru)
        for K in (0.05, 0.25, 1.0, np.pi, 7.0):
            f = uD.laplacian() + uD * (K * K)

            def bc(t):
                return utru(t, jnp.ones_like(t))

            u = Diskfun.helmholtz(f, K, bc, m=120)
            assert _nrm(u, utru) < 5e4 * TOL, K
