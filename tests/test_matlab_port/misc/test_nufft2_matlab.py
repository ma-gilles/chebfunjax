"""Port of MATLAB Chebfun tests/misc/test_nufft2.m (Fable 5).

Native dimensions and matrix spectral norms are retained. MATLAB rng(0)
maps to numpy.default_rng(0); random streams differ.

Provenance
----------
MATLAB source : tests/misc/test_nufft2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

import chebfunjax as cj


class TestNufft2:
    def test_matches_direct_sum(self):
        rng = np.random.default_rng(0)
        M, N = 100, 103
        m, n = 105, 101
        C = rng.random((m, n)) + 1j * rng.random((m, n))
        x = rng.random((M, N))
        y = rng.random((M, N))
        F = np.asarray(cj.nufft2(C, x, y))
        Y = np.zeros((M, N), dtype=complex)
        for j in range(M):
            for k in range(N):
                Y[j, k] = (
                    np.exp(-2j * np.pi * y[j, k] * np.arange(m))
                    @ C
                    @ np.exp(-2j * np.pi * x[j, k] * np.arange(n))
                )
        assert (
            np.linalg.norm(F - Y, ord=2) / np.linalg.norm(C, ord=2)
            < 10 * M * N * np.finfo(float).eps
        )
