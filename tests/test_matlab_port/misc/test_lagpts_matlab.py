"""Port of MATLAB Chebfun tests/misc/test_lagpts.m (Fable 5).

The Python API returns all native quadrature vectors as one-dimensional
arrays; MATLAB's row/column distinction is checked here as the corresponding
length contract.

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
    def test_n42_gq_source_assertions(self):
        # Source pass(1): x-only output has n entries.
        x_only, _ = lagpts(42)
        assert x_only.shape == (42,)

        # Source pass(2): x, w, v all have n entries in the native 1D API.
        x, w, v = lagpts(42, bary=True)
        assert x.shape == (42,)
        assert w.shape == (42,)
        assert v.shape == (42,)

        # Source pass(3): first and second moments.
        assert abs(np.dot(w, x) - 1.0) <= TOL
        assert abs(np.dot(w, x**2) - 2.0) <= TOL

        # Source pass(4-6): selected node, quadrature weight, and baryweight.
        assert abs(x[36] - 98.388267163326702) < TOL
        assert abs(w[6] - 0.055372813167092) < TOL
        assert abs(v[16] - 0.002937421407003) < TOL

    def test_n251_asy_source_assertions(self):
        # Source pass(7): x-only output has n entries.
        x_only, _ = lagpts(251)
        assert x_only.shape == (251,)

        # Source pass(8): all three output vectors have n entries.
        x, w, v = lagpts(251, bary=True)
        assert x.shape == (251,)
        assert w.shape == (251,)
        assert v.shape == (251,)

        # Source pass(9-12): moments and selected ASY values.
        assert abs(np.dot(w, x) - 1.0) < 100 * TOL
        assert abs(np.dot(w, x**2) - 2.0) < 100 * TOL
        assert abs(x[36] - 13.309000189442097) < TOL
        assert abs(w[2] - 0.050091759039996) < TOL
        assert abs(v[2] - 0.214530194346947) < 10 * TOL

    def test_positive_semi_infinite_shifted_interval(self):
        # Source pass(13): lagpts(n, [1, inf]).
        x, w, _ = lagpts(42, interval=(1.0, np.inf), bary=True)
        assert abs(np.dot(w, x) - 2.0 / np.e) < TOL
        assert abs(np.dot(w, x**2) - 5.0 / np.e) < TOL

    def test_negative_semi_infinite_shifted_interval(self):
        # Source pass(14): lagpts(n, [-inf, -1]).
        x, w, _ = lagpts(42, interval=(-np.inf, -1.0), bary=True)
        assert abs(np.dot(w, x) + 2.0 / np.e) < TOL
        assert abs(np.dot(w, x**2) - 5.0 / np.e) < TOL
