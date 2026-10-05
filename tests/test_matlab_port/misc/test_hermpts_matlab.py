"""Port of MATLAB Chebfun tests/misc/test_hermpts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_hermpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

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
        assert abs(float(np.asarray(v)[16]) - 0.311886101735772) < TOL

    def test_n251_moments(self):
        x, w, v = hermpts(251, bary=True)
        x, w, v = np.asarray(x), np.asarray(w), np.asarray(v)
        assert x.shape == (251,)
        assert w.shape == v.shape == (251,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - np.sqrt(np.pi) / 2) < 300 * TOL
        assert abs(x[36] + 13.292221459334638) < 4 * TOL
        assert abs(w[122] - 0.117419270715955) < 10 * TOL
        assert abs(v[122] - 0.915560323259764) < 100 * TOL

    def test_probabilist_scaling_and_explicit_phys(self):
        # MATLAB pass(12)-(15).
        x, w, v = hermpts(251, bary=True)
        x2, w2, v2 = hermpts(251, 'prob', bary=True)
        assert np.max(np.abs(np.asarray(x - x2 / np.sqrt(2)))) < TOL
        assert np.max(np.abs(np.asarray(w - w2 / np.sqrt(2)))) < TOL
        np.testing.assert_array_equal(v, v2)
        x3, w3 = hermpts(251, 'phys')
        np.testing.assert_array_equal(x, x3)
        np.testing.assert_array_equal(w, w3)

    @pytest.mark.parametrize('n', [1, 2, 19, 42, 251])
    def test_explicit_glr(self, n):
        x, w, v = map(np.asarray, hermpts(n, 'GLR', bary=True))
        assert len(x) == len(w) == len(v) == n
        np.testing.assert_array_equal(x, -x[::-1])
        np.testing.assert_array_equal(w, w[::-1])
        assert abs(w.sum() - np.sqrt(np.pi)) < TOL
        if n > 1:
            assert abs(w @ x**2 - np.sqrt(np.pi) / 2) < 300 * TOL

    @pytest.mark.parametrize('n', [0, 1])
    def test_empty_and_singleton(self, n):
        x, w, v = map(np.asarray, hermpts(n, bary=True))
        assert x.shape == w.shape == v.shape == (n,)
        if n:
            np.testing.assert_array_equal(x, [0.0])
            np.testing.assert_array_equal(v, [1.0])

    @pytest.mark.parametrize('n', [-1, 2.5])
    def test_invalid_order(self, n):
        with pytest.raises(ValueError, match='nonnegative integer'):
            hermpts(n)

    @pytest.mark.parametrize('n', [21, 42, 99, 198, 199])
    def test_rec_default_range_against_independent_rule(self, n):
        # NumPy's polynomial quadrature is independent of the ported
        # Airy/Tricomi seeds and scaled Hermite recurrence.
        x, w, v = map(np.asarray, hermpts(n, 'REC', bary=True))
        expected_x, expected_w = np.polynomial.hermite.hermgauss(n)
        np.testing.assert_allclose(x, expected_x, atol=100 * EPS, rtol=100 * EPS)
        np.testing.assert_allclose(w, expected_w, atol=100 * EPS, rtol=100 * EPS)
        np.testing.assert_array_equal(x, -x[::-1])
        np.testing.assert_array_equal(w, w[::-1])
        assert abs(w.sum() - np.sqrt(np.pi)) < TOL
        assert abs(w @ x**2 - np.sqrt(np.pi) / 2) < TOL
        np.testing.assert_array_equal(np.sign(v), (-1.0) ** np.arange(n))
        for actual, default in zip((x, w, v), hermpts(n, bary=True), strict=True):
            np.testing.assert_array_equal(actual, default)
