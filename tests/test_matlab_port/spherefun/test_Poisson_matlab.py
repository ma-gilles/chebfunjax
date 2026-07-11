"""Port of MATLAB Chebfun tests/spherefun/test_Poisson.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_Poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-9


class TestSpherefunPoisson:
    def test_recovers_harmonic(self):
        # solve lap u = -l(l+1) Y_l^m  ->  u = Y_l^m (zero-mean gauge)
        l, m = 3, 1
        Y = Spherefun.sphharm(l, m)
        rhs = Spherefun.from_function(
            lambda lam, th: -l * (l + 1) * Y(lam, th))
        u = Spherefun.poisson(rhs, const=0.0)
        lam, th = jnp.asarray(0.8), jnp.asarray(0.9)
        assert abs(float(u(lam, th)) - float(Y(lam, th))) < TOL
