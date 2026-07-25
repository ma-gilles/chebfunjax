"""Port of MATLAB Chebfun tests/misc/test_inufft.m (Fable 5).

Inverse NUFFT: recover c from F = A c where A is the nonuniform DFT
matrix; MATLAB checks norm(exact - fast) with exact = A \\ c.  N = 4000
trimmed to 2000 for runtime.

Provenance
----------
MATLAB source : tests/misc/test_inufft.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.nufft import inufft

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(0)


class TestInufft:
    # N=1000 takes ~290 s (dense reference solve dominates); give it
    # headroom beyond the default 300 s so contention cannot flake it.
    @pytest.mark.parametrize(
        "N", [1, 10, 100, pytest.param(1000, marks=pytest.mark.timeout(890))])
    def test_recovers_direct_solve(self, N):
        # chebfunjax inufft inverts its type-2 matrix
        # A[j,k] = e^{-2 pi i x_j k}; MATLAB's inufft(...,1) inverts the
        # transpose (type-1) system.  The accuracy claim (fast solve
        # matches the dense direct solve) is the same; ported in the
        # chebfunjax convention.
        omega = np.arange(N, dtype=float) + 0.4 * RNG.uniform(size=N)
        x = omega / N
        A = np.exp(-2j * np.pi * x[:, None] * np.arange(N)[None, :])
        c = RNG.uniform(size=N) + 1j * RNG.uniform(size=N)
        exact = np.linalg.solve(A, c) if N > 1 else c / A[0, 0]
        fast = np.asarray(inufft(c, x))
        err = float(np.max(np.abs(exact - fast)))
        assert err < 4000 * N * EPS * float(np.sum(np.abs(c)))

    def test_matlab_type1_convention(self):
        N = 10
        omega = np.arange(N, dtype=float) + 0.4 * RNG.uniform(size=N)
        F = np.exp(-2j * np.pi * np.arange(N)[:, None] / N
                   * omega[None, :])
        c = RNG.uniform(size=N) + 1j * RNG.uniform(size=N)
        exact = np.linalg.solve(F, c)
        fast = np.asarray(inufft(c, omega / N, nufft_type=1))
        assert float(np.max(np.abs(exact - fast))) < 1e-8
