"""Port of MATLAB Chebfun tests/misc/test_radaupts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_radaupts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from chebfunjax.utils.quadrature import radaupts

TOL = 1e-14


class TestRadaupts:
    def test_n42_moments_and_references(self):
        x, w = radaupts(42)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL
        assert abs(x[36] - 0.908847278001044) < TOL
        assert abs(w[36] - 0.031190846817016) < TOL
        assert x[0] == -1.0

    def test_barycentric(self):
        # FIXED (Fable 5): bary=True returns MATLAB's third output.
        x, w, v = radaupts(42, bary=True)
        assert abs(float(np.asarray(v)[36]) + 0.171069152683909) < 1e-12

    def test_jacobi_radau_exactness(self):
        n, alp, bet = 8, 0.3, 1.2
        x, w = radaupts(n, alp, bet)
        x, w = np.asarray(x), np.asarray(w)
        for k in range(2 * n - 2):
            exact, _ = quad(
                lambda t, k=k: t ** k * (1 - t) ** alp * (1 + t) ** bet,
                -1, 1, epsabs=1e-13, epsrel=1e-13)
            assert abs(np.dot(w, x ** k) - exact) < 1e-12, f"k={k}"
