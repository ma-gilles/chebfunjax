"""Port of MATLAB Chebfun tests/ballfun/test_helmholtz.m (Fable 5).

FIXED: Ballfun.helmholtz added in the Fable 5 audit (spectral
spherical-harmonic x Chebyshev-collocation solve with Dirichlet data).

Provenance
----------
MATLAB source : tests/ballfun/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e7 * np.finfo(float).eps
RS = jnp.asarray(np.linspace(0.05, 1.0, 7))
LAMS = jnp.asarray(np.linspace(-3, 3, 7))
THS = jnp.asarray(np.linspace(0.1, 3.0, 7))
RR, LL, TT = jnp.meshgrid(RS, LAMS, THS, indexing="ij")


def _nrm(u, ex):
    return float(jnp.max(jnp.abs(u(RR, LL, TT) - ex(RR, LL, TT))))


class TestBallfunHelmholtz:
    def test_poisson_limit(self):
        # pass(1): u = 1
        u = Ballfun.helmholtz(lambda r, l, t: 0 * r, 0.0,
                              lambda l, t: 1.0 + 0 * l)
        assert _nrm(u, lambda r, l, t: 1.0 + 0 * r) < TOL

        # pass(2): u = r^2
        u = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0 * r, 0.0,
                              lambda l, t: 1.0 + 0 * l)
        assert _nrm(u, lambda r, l, t: r ** 2) < TOL

        # pass(3): u = r^2 sin(th)^2
        u = Ballfun.helmholtz(lambda r, l, t: 4.0 + 0 * r, 0.0,
                              lambda l, t: jnp.sin(t) ** 2)
        assert _nrm(u, lambda r, l, t: r ** 2 * jnp.sin(t) ** 2) < TOL

    def test_nonzero_k(self):
        # manufactured: u = r^2 sin(th)^2, lap u = 4, f = 4 + K^2 u
        K = 2.0
        u = Ballfun.helmholtz(
            lambda r, l, t: 4.0 + K * K * (r ** 2 * jnp.sin(t) ** 2),
            K, lambda l, t: jnp.sin(t) ** 2)
        assert _nrm(u, lambda r, l, t: r ** 2 * jnp.sin(t) ** 2) < TOL
