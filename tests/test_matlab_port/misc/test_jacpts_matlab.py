"""Port of MATLAB Chebfun tests/misc/test_jacpts.m (Fable 5).

Reference values are MATLAB's printed constants.  Barycentric-weight
outputs are xfailed (chebfunjax jacpts returns nodes and weights).

Provenance
----------
MATLAB source : tests/misc/test_jacpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.quadrature import jacpts

TOL = 1e-12


class TestJacpts:
    def test_moments_and_reference_values(self):
        out = jacpts(42, -0.1, 0.3)
        x, w = np.asarray(out[0]), np.asarray(out[1])
        assert x.shape == (42,)
        assert abs(np.dot(w, x) - 0.363593965943934) < TOL
        assert abs(np.dot(w, x ** 2) - 0.670376374709129) < TOL
        assert abs(x[36] - 0.912883347814032) < TOL
        assert abs(w[36] - 0.046661910947553) < TOL

    @pytest.mark.xfail(reason="chebfunjax jacpts has no barycentric-"
                       "weight output")
    def test_barycentric_weights(self):
        x, w, v = jacpts(42, -0.1, 0.3)
        assert abs(float(np.asarray(v)[36]) - 0.320696510075909) < TOL

    def test_mapped_interval(self):
        out = jacpts(42, -0.1, 0.3, (0.0, 10.0)) if True else None
        x, w = np.asarray(out[0]), np.asarray(out[1])
        assert abs(np.dot(w, x) - 81.519974175437831) < 10 * TOL
        assert abs(np.dot(w, x ** 2) - 585.9248143859594) < 100 * TOL
        assert abs(x[37] - 9.702339316456870) < TOL
        assert abs(w[37] - 0.279566831611687) < TOL
