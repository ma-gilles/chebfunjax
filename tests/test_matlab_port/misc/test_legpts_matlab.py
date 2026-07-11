"""Port of MATLAB Chebfun tests/misc/test_legpts.m (Fable 5).

chebfunjax legpts returns (x, w) without the barycentric weights v;
those assertions are xfailed.  Reference node/weight values are
MATLAB's own printed constants.

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
        x, w = legpts(42)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert w.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL

    def test_n42_reference_values(self):
        x, w = legpts(42)
        assert abs(float(np.asarray(x)[36]) - 0.910959724904127) < TOL
        assert abs(float(np.asarray(w)[36]) - 0.030479240699603) < TOL

    @pytest.mark.xfail(reason="chebfunjax legpts returns (x, w) without "
                       "barycentric weights v")
    def test_n42_barycentric_weights(self):
        x, w, v = legpts(42)  # noqa
        assert abs(float(np.asarray(v)[36]) - 0.265155501739424) < TOL

    def test_mapped_interval(self):
        x, w = legpts(42, (0.0, 10.0))
        x, w = np.asarray(x), np.asarray(w)
        assert abs(np.dot(w, x) - 50) < 10 * TOL
        assert abs(np.dot(w, x ** 2) - 1000 / 3) < 100 * TOL
        assert abs(x[37] - 9.694617786774941) < TOL
        assert abs(w[37] - 0.127114797630565) < TOL

    def test_n251_moments(self):
        # n=251 uses the asymptotic/Newton path in MATLAB; same check.
        x, w = legpts(251)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (251,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 2 / 3) < TOL

    def test_large_n_moments(self):
        x, w = legpts(1013)
        x, w = np.asarray(x), np.asarray(w)
        assert abs(np.dot(w, x)) < 1e-13
        assert abs(np.dot(w, x ** 2) - 2 / 3) < 1e-13
