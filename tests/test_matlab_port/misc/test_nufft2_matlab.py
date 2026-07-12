"""Port of MATLAB Chebfun tests/misc/test_nufft2.m (Fable 5).

FIXED: nufft2 added in the Fable 5 audit.  (Smaller sizes than the
MATLAB test's 100x103 for runtime; the assertion is identical --
transform matches the direct double sum.)

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
        M, N = 40, 43
        m, n = 45, 41
        C = rng.random((m, n)) + 1j * rng.random((m, n))
        x = rng.random((M, N))
        y = rng.random((M, N))
        F = np.asarray(cj.nufft2(C, x, y))
        Y = np.zeros((M, N), dtype=complex)
        for j in range(M):
            for k in range(N):
                Y[j, k] = (np.exp(-2j * np.pi * y[j, k]
                                  * np.arange(m)) @ C
                           @ np.exp(-2j * np.pi * x[j, k]
                                    * np.arange(n)))
        assert np.linalg.norm(F - Y) / np.linalg.norm(C) \
            < 10 * M * N * np.finfo(float).eps
