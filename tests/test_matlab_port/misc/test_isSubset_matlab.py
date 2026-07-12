"""Port of MATLAB Chebfun tests/misc/test_isSubset.m (Fable 5).

FIXED: isSubset added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/misc/test_isSubset.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

import chebfunjax as cj

TOL = 100 * np.finfo(float).eps


class TestIsSubset:
    def test_1d_domains(self):
        assert cj.isSubset([0, 2], [0, 2], TOL)
        assert not cj.isSubset([0, 2], [0, 1], TOL)
        assert cj.isSubset([-0.7, -0.5], [-1, 1], TOL)
        # not exactly contained, but within tolerance
        eps = np.finfo(float).eps
        assert cj.isSubset([-eps, 1], [0, 1], TOL)

    def test_2d_domains(self):
        assert cj.isSubset([-0.7, -0.5, 1, 2], [-1, 1, -1, 3], TOL)
        assert not cj.isSubset([-0.7, -0.5, 1, 2], [-1, 1, -1, 1],
                               TOL)

    def test_3d_domains(self):
        A = [0, 1, 0, 1, 0, 1]
        assert cj.isSubset(A, [-1, 2, -1, 2, -1, 2], TOL)
        assert not cj.isSubset(A, [-1, 2, -1, 2, 0.5, 2], TOL)
