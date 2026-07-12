"""Port of MATLAB Chebfun tests/ballfun/test_solharm.m (Fable 5).

FIXED: Ballfun.solharm added in the Fable 5 audit
(sqrt(2l+3) r^l Y_lm, unit ball L2 norm).  l runs to 4 rather than
MATLAB's 10 for runtime; every assertion is the same.

Provenance
----------
MATLAB source : tests/ballfun/test_solharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e6 * np.finfo(float).eps
LAMS = jnp.asarray(np.linspace(-3, 3, 9))
THS = jnp.asarray(np.linspace(0.1, 3.0, 9))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")
RS = jnp.asarray(np.linspace(0.05, 1.0, 6))
RR3, LL3, TT3 = jnp.meshgrid(
    RS, jnp.asarray(np.linspace(-3, 3, 6)),
    jnp.asarray(np.linspace(0.1, 3.0, 6)), indexing="ij")


class TestBallfunSolharm:
    def test_surface_norm_laplacian(self):
        maxd = maxn = maxlap = 0.0
        for l in range(5):
            for m in range(-l, l + 1):
                Y = Ballfun.solharm(l, m)
                Z = Spherefun.sphharm(l, m)
                # surface restriction / sqrt(2l+3) equals sphharm
                d = float(jnp.max(jnp.abs(
                    Y(jnp.ones_like(LL), LL, TT)
                    / np.sqrt(2 * l + 3) - Z(LL, TT))))
                maxd = max(maxd, d)
                # unit ball L2 norm
                maxn = max(maxn, abs(float(Y.norm()) - 1))
                # harmonic: laplacian == 0
                maxlap = max(maxlap, float(jnp.max(jnp.abs(
                    Y.laplacian()(RR3, LL3, TT3)))))
        assert maxd < TOL
        assert maxn < TOL
        assert maxlap < TOL

    def test_y00(self):
        Y = Ballfun.solharm(0, 0)
        exact = 0.5 * np.sqrt(1 / np.pi) * np.sqrt(3)
        assert float(jnp.max(jnp.abs(
            Y(RR3, LL3, TT3) - exact))) < TOL
