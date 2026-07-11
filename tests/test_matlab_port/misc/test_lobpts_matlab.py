"""Port of MATLAB Chebfun tests/misc/test_lobpts.m (Fable 5).

The Jacobi-Lobatto (alp, bet) exactness sweep checks quadrature
exactness for monomials against scipy reference integrals.

Provenance
----------
MATLAB source : tests/misc/test_lobpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from chebfunjax.utils.quadrature import lobpts

TOL = 1e-14


class TestLobpts:
    def test_n42_moments_and_references(self):
        x, w = lobpts(42)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL
        assert abs(x[36] - 0.922259214258616) < TOL
        assert abs(w[36] - 0.029306411216166) < TOL
        assert x[0] == -1.0 and x[-1] == 1.0

    @pytest.mark.xfail(reason="chebfunjax lobpts has no barycentric-"
                       "weight output")
    def test_barycentric(self):
        x, w, v = lobpts(42)
        assert abs(float(np.asarray(v)[36]) + 0.622355798366776) < TOL

    def test_jacobi_lobatto_exactness(self):
        n, alp, bet = 8, 0.3, 1.2
        x, w = lobpts(n, alp, bet)
        x, w = np.asarray(x), np.asarray(w)
        for k in range(2 * n - 3):
            exact, _ = quad(
                lambda t, k=k: t ** k * (1 - t) ** alp * (1 + t) ** bet,
                -1, 1, epsabs=1e-13, epsrel=1e-13)
            assert abs(np.dot(w, x ** k) - exact) < 1e-12, f"k={k}"
