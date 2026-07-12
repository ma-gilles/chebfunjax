"""Port of MATLAB Chebfun tests/spherefun/test_gaussfilt.m (Fable 5).

FIXED: Spherefun.gaussfilt added in the Fable 5 audit (one
backward-Euler heat step on spherical-harmonic coefficients).

Provenance
----------
MATLAB source : tests/spherefun/test_gaussfilt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e3 * np.finfo(float).eps
LAMS = jnp.asarray(np.linspace(-3, 3, 11))
THS = jnp.asarray(np.linspace(0.1, 3.0, 11))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")


class TestSpherefunGaussfilt:
    def test_constant_unchanged(self):
        # pass(1)-(2)
        f = Spherefun.from_function(lambda lam, th: 1.0 + 0 * lam)
        for sig in (None, 2):
            g = f.gaussfilt() if sig is None else f.gaussfilt(sig)
            assert float(jnp.max(jnp.abs(g(LL, TT) - 1.0))) < TOL

    def test_norm_decreases(self):
        # pass(3)-(5)
        f = Spherefun.sphharm(13, 7)
        prev = float(f.norm())
        for _ in range(3):
            g = f.gaussfilt(100)
            cur = float(g.norm())
            assert cur < prev
            prev, f = cur, g

    def test_mean_preserved(self):
        # pass(6): mean of 2 + Y_12^5 is 2
        f = Spherefun.from_function(
            lambda lam, th: 2.0 + Spherefun.sphharm(12, 5)(lam, th))
        g = f.gaussfilt(2)
        assert abs(float(g.sum2()) / (4 * np.pi) - 2) < TOL
