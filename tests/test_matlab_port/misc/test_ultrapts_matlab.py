"""Port of MATLAB Chebfun tests/misc/test_ultrapts.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_ultrapts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.quadrature import ultrapts

TOL = 1e-14


class TestUltrapts:
    def test_n42_moments_and_references(self):
        x, w = ultrapts(42, 0.3)
        x, w = np.asarray(x), np.asarray(w)
        assert x.shape == (42,)
        assert abs(np.dot(w, x)) < TOL
        assert abs(np.dot(w, x ** 2) - 0.8843414686338345) < TOL
        assert abs(x[36] - 9.131896381993957e-01) < TOL
        assert abs(w[36] - 4.332670514309510e-02) < TOL

    def test_barycentric(self):
        # FIXED (Fable 5): bary=True returns MATLAB's third output.
        x, w, v = ultrapts(42, 0.3, bary=True)
        assert abs(float(np.asarray(v)[36]) - 3.115587460502451e-01) < 1e-11

    def test_mapped_interval(self):
        x, w = ultrapts(42, 0.3, (0.0, 10.0))
        x, w = np.asarray(x), np.asarray(w)
        assert abs(np.dot(w, x) - 30.1957169274024) < 31 * TOL
        assert abs(np.dot(w, x ** 2) - 209.0472710358626) < 300 * TOL
        assert abs(x[37] - 9.704505777068543) < TOL
        assert abs(w[37] - 1.018229378664342e-01) < TOL
