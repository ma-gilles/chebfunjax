"""Port of MATLAB Chebfun tests/spherefun/test_HelmholtzSolver.m
(Fable 5).

FIXED: spherefun.helmholtz added in the Fable 5 audit (spectral
spherical-harmonic solve).

Provenance
----------
MATLAB source : tests/spherefun/test_HelmholtzSolver.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-10
LAMS = jnp.asarray(np.linspace(-3, 3, 11))
THS = jnp.asarray(np.linspace(0.1, 3.0, 11))
LL, TT = jnp.meshgrid(LAMS, THS, indexing="ij")


class TestSpherefunHelmholtz:
    def test_sphharm_eigenfunctions(self):
        K = 100.1
        for L in range(4):
            for M in range(L + 1):
                f = Spherefun.sphharm(L, M)
                u = Spherefun.helmholtz(
                    lambda lam, th: (K ** 2 - L * (L + 1))
                    * f(lam, th), K)
                assert float(jnp.max(jnp.abs(
                    u(LL, TT) - f(LL, TT)))) < 100 * TOL, (L, M)
