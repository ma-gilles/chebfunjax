"""Port of MATLAB Chebfun tests/misc/test_lagpts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_lagpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.quadrature import lagpts

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS


class TestLagpts:
    def test_n42_moments_and_references(self):
        x, w = lagpts(42)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert abs(np.dot(w, x) - 1) <= TOL
        assert abs(np.dot(w, x ** 2) - 2) <= TOL
        assert abs(x[36] - 98.388267163326702) < TOL * 100
        assert abs(w[6] - 0.055372813167092) < TOL

    def test_barycentric(self):
        # FIXED (Fable 5): bary=True returns MATLAB's third output.
        x, w, v = lagpts(42, bary=True)
        assert abs(float(np.asarray(v)[16]) - 0.002937421407003) < 1e-12

    def test_n251_first_moment(self):
        x, w = lagpts(251)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (251,)
        assert abs(np.dot(w, x) - 1) < 1e-10
