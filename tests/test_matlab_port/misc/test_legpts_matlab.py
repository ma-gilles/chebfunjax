"""Port of MATLAB Chebfun tests/misc/test_legpts.m (Fable 5).

All thirty numerical assertions are retained, with Python one-dimensional
arrays for MATLAB row/column vectors. Reference node, weight and barycentric
values are MATLAB's own printed constants.

Provenance
----------
MATLAB source : tests/misc/test_legpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.quadrature import legpts

TOL = 1e-14


class TestLegpts:
    def test_n42_shapes_and_moments(self):
        x, w, v = legpts(42, bary=True)
        x, w, v = np.asarray(x), np.asarray(w), np.asarray(v)
        assert x.shape == (42,)
        assert w.shape == (42,)
        assert v.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL

    def test_n42_reference_values(self):
        x, w = legpts(42)
        assert abs(float(np.asarray(x)[36]) - 0.910959724904127) < TOL
        assert abs(float(np.asarray(w)[36]) - 0.030479240699603) < TOL

    def test_n42_barycentric_weights(self):
        # FIXED (Fable 5): bary=True returns MATLAB's third output.
        x, w, v = legpts(42, bary=True)
        assert abs(float(np.asarray(v)[36]) - 0.265155501739424) < TOL

    def test_mapped_interval(self):
        x, w, v = legpts(42, (0.0, 10.0), bary=True)
        x, w = np.asarray(x), np.asarray(w)
        assert abs(np.dot(w, x) - 50) < 10 * TOL
        assert abs(np.dot(w, x ** 2) - 1000 / 3) < 100 * TOL
        assert abs(x[37] - 9.694617786774941) < TOL
        assert abs(w[37] - 0.127114797630565) < TOL
        assert abs(float(v[37]) + 0.202027188941007) < TOL

    def test_n251_moments(self):
        # n=251 uses the asymptotic/Newton path in MATLAB; same check.
        x, w, v = legpts(251, bary=True)
        x, w, v = np.asarray(x), np.asarray(w), np.asarray(v)
        assert x.shape == (251,)
        assert w.shape == v.shape == (251,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL
        assert abs(x[36] + 0.896467746955729) < TOL
        assert abs(w[36] - 0.005535005742012) < TOL
        assert abs(v[36] - 0.294960654628873) < TOL

    def test_n251_mapped_reference_values(self):
        x, w, v = (np.asarray(a) for a in legpts(251, (0.0, 10.0), bary=True))
        assert abs(np.dot(w, x) - 50) < 10 * TOL
        assert abs(np.dot(w, x**2) - 1000 / 3) < 100 * TOL
        assert abs(x[37] - 0.545685271938239) < TOL
        assert abs(w[37] - 0.028372255931272) < TOL
        assert abs(v[37] + 0.306176997099458) < TOL

    @pytest.mark.parametrize('n', [1, 2])
    def test_trivial_rules(self, n):
        x, w = (np.asarray(a) for a in legpts(n))
        if n == 1:
            assert x[0] == 0 and w[0] == 2
        else:
            np.testing.assert_allclose(x, [-1 / np.sqrt(3), 1 / np.sqrt(3)], atol=TOL, rtol=0)
            np.testing.assert_allclose(w, [1, 1], atol=TOL, rtol=0)
        x, w = (np.asarray(a) for a in legpts(n, (-10.0, 3.0)))
        assert abs(np.dot(w, x) + 45.5) < TOL
        if n == 1:
            assert abs((x[0] + 10) + (x[0] - 3)) < TOL
        else:
            assert abs(w.sum() - 13) < TOL
            assert abs(np.dot(w, x**2) - (342 + 1 / 3)) < 20 * TOL
            assert abs(np.dot(w, x**3) + 2479.75) < 1000 * TOL

    def test_large_n_moments(self):
        x, w = legpts(1013)
        x, w = np.asarray(x), np.asarray(w)
        assert abs(np.dot(w, x)) < 1e-13
        assert abs(np.dot(w, x ** 2) - 2 / 3) < 1e-13
