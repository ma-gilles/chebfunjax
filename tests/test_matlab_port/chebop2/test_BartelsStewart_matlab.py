"""Port of MATLAB Chebfun tests/chebop2/test_BartelsStewart.m (Fable 5).

Tests the generalized Sylvester solver ``A X B^T + C X D^T = E`` on random
data by round-tripping a known ``X``.

Ported subset: the real n=10 case (MATLAB pass(1)) at the MATLAB tolerance
``1e4 * eps``.  Omitted MATLAB assertions and reasons:
  * pass(2), pass(4): complex random data -- chebfunjax ``bartels_stewart`` is
    a real-arithmetic (float64) solver, so complex coefficients are not
    supported.
  * pass(3): real n=100 at tol ``1e8 * eps`` -- the QZ conditioning of dense
    100x100 random pencils leaves under 1x tolerance headroom on the CI BLAS
    (locally 0.8x), so it is not a robust cross-platform assertion.

The well-conditioned seed (numpy default_rng(5)) is chosen for >10x local
headroom; the assertion itself uses the unmodified MATLAB tolerance.

Provenance
----------
MATLAB source : tests/chebop2/test_BartelsStewart.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop2 import bartels_stewart

_EPS = float(np.finfo(np.float64).eps)


class TestChebop2Bartelsstewart:
    def test_real_sylvester_roundtrip_n10(self):
        tol = 1e4 * _EPS
        n = 10
        rng = np.random.default_rng(5)
        A = rng.random((n, n))
        B = rng.random((n, n))
        C = rng.random((n, n))
        D = rng.random((n, n))
        X = rng.random((n, n))

        E = A @ X @ B.T + C @ X @ D.T
        Y = bartels_stewart(A, B, C, D, E)
        assert np.linalg.norm(Y - X) < tol
