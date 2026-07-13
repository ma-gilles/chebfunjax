"""Port of MATLAB Chebfun tests/ballfun/test_mean.m (Fable 5).

FIXED: Ballfun.mean(dim) added in the Fable 5 audit (dim=1 averages
over r -> Spherefun; dim=2/3 average over lambda/theta -> Diskfun
with the doubled-colatitude convention).

Provenance
----------
MATLAB source : tests/ballfun/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e-12
LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 7))
THS = jnp.asarray(np.linspace(0.1, 3.0, 7))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")
TD = jnp.asarray(np.linspace(-np.pi, np.pi, 7))
RD = jnp.asarray(np.linspace(0.0, 1.0, 5))
T2, R2 = jnp.meshgrid(TD, RD, indexing="ij")


class TestBallfunMean:
    def test_constant_means(self):
        # pass(1)-(3): constant functions average to themselves
        f1 = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert float(jnp.max(jnp.abs(f1.mean(1)(LL, TT) - 1.0))) \
            < TOL
        f2 = Ballfun.from_function(lambda x, y, z: 2.0 + 0 * x)
        assert float(jnp.max(jnp.abs(f2.mean(2)(T2, R2) - 2.0))) \
            < TOL
        f3 = Ballfun.from_function(lambda x, y, z: 3.0 + 0 * x)
        assert float(jnp.max(jnp.abs(f3.mean(3)(T2, R2) - 3.0))) \
            < TOL

    def test_radial_mean(self):
        # mean over r of r^2 is 1/3 on the sphere
        f = Ballfun.from_function(lambda r, lam, th: r ** 2,
                                  spherical=True)
        assert float(jnp.max(jnp.abs(
            f.mean(1)(LL, TT) - 1.0 / 3.0))) < TOL
