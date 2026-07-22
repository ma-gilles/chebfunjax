"""Port of MATLAB Chebfun tests/chebop/test_eigs_periodic.m (Fable 5).

Periodic eigenvalue problems solved by Fourier (trig) collocation:
``eigs(L, k)`` with ``L.bc = 'periodic'``.

Provenance
----------
MATLAB source : tests/chebop/test_eigs_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chebfunjax.operators.chebop import Chebop  # noqa: E402
from chebfunjax.tech.trigtech import Trigtech  # noqa: E402

# tol = 1e2 * bvpTol, bvpTol = 5e-13 (cheboppref factory default).
TOL = 1e2 * 5e-13


class TestChebopEigsPeriodic:
    def test_laplacian_periodic(self):
        # -u'' = lambda u on [0, 2*pi], periodic BCs.
        L = Chebop(lambda u: -u.diff(2), (0.0, 2 * np.pi))
        L.bc = "periodic"
        V, D = L.eigs(k=7, return_eigenfunctions=True)
        D = np.sort(np.real(np.asarray(D)))
        Dexact = np.array([0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0])
        assert np.max(np.abs(D - Dexact)) < TOL
        # The eigenfunctions are trigonometric (MATLAB checks tech == trigtech).
        assert isinstance(V[0].funs[0].tech, Trigtech)

    def test_mathieu_periodic(self):
        # -u'' + 2*q*cos(2x) u = lambda u on [0, 2*pi], periodic BCs.
        q = 2.0
        L = Chebop(lambda x, u: -u.diff(2) + 2 * q * jnp.cos(2 * x) * u,
                   (0.0, 2 * np.pi))
        L.bc = "periodic"
        D = np.sort(np.real(np.asarray(L.eigs(k=7))))
        Dwolfram = np.array([
            -1.513956885056520,
            -1.390676501225323,
            2.379199880488686,
            3.672232706497191,
            5.172665133358294,
            9.140627737766440,
            9.370322483621104,
        ])
        assert np.max(np.abs(D - Dwolfram)) < TOL
