"""Port of MATLAB Chebfun tests/spherefun/test_HelmholtzSolver.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_HelmholtzSolver.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import pytest

from chebfunjax.spherefun.spherefun import Spherefun

jax.config.update("jax_enable_x64", True)


class TestSpherefunHelmholtzSolver:
    @pytest.mark.parametrize("m,n", [(60, 40), (61, 41), (62, 42), (63, 43)])
    def test_sphharm_eigenfunctions(self, m, n):
        tol = 1e-10
        K = 100.1
        for L in range(4):
            for M in range(L + 1):
                f = Spherefun.sphharm(L, M)
                u = Spherefun.helmholtz((K ** 2 - L * (L + 1)) * f, K, m, n)
                assert float((u - f).norm(2)) < 100 * tol, (L, M)
