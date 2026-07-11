"""Port of MATLAB Chebfun tests/misc/test_hermpts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_hermpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.quadrature import hermpts

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS


class TestHermpts:
    def test_n42_moments_and_references(self):
        x, w = hermpts(42)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - np.sqrt(np.pi) / 2) < TOL
        assert abs(x[36] - 5.660357581283058) < 10 * TOL
        assert abs(w[16] - 0.032202101288908) < TOL

    def test_barycentric(self):
        # FIXED (Fable 5): bary=True returns MATLAB's third output.
        x, w, v = hermpts(42, bary=True)
        assert abs(float(np.asarray(v)[16]) - 0.311886101735772) < 1e-12

    def test_n251_moments(self):
        x, w = hermpts(251)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (251,)
        assert abs(np.dot(w, x)) < 100 * TOL
        assert abs(np.dot(w, x ** 2) - np.sqrt(np.pi) / 2) < 100 * TOL
