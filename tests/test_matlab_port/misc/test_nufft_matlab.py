"""Port of MATLAB Chebfun tests/misc/test_nufft.m (Fable 5).

Type-1 NUFFT vs the direct nonuniform DFT at MATLAB's tolerance
300*N*eps*norm(c,1), for N = 10^0..10^3 (10^4 trimmed for CI runtime).

Provenance
----------
MATLAB source : tests/misc/test_nufft.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.nufft import nufft

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(0)


def nudft1(c, omega):
    n = len(c)
    k = np.arange(n)
    return np.exp(-2j * np.pi * omega[:, None] * k[None, :] / n) @ c


class TestNufft:
    @pytest.mark.parametrize("N", [1, 10, 100, 1000])
    def test_type1_matches_direct(self, N):
        omega = np.arange(N, dtype=float) + 1.2 * RNG.uniform(size=N) / N
        c = RNG.uniform(size=N) + 1j * RNG.uniform(size=N)
        exact = nudft1(c, omega)
        # Convention map: MATLAB's type-1 (F_j = sum_k c_k
        # e^{-2pi i omega_j k / N}) is chebfunjax's type-2 evaluated at
        # x = omega/N (the two libraries' type-1/2 are adjoint pairs).
        fast = np.asarray(nufft(c, omega / N, nufft_type=2))
        err = float(np.max(np.abs(exact - fast)))
        assert err < 300 * N * EPS * float(np.sum(np.abs(c)))
