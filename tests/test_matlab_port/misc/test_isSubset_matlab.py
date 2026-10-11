"""Port of MATLAB Chebfun tests/misc/test_isSubset.m (Fable 5).

All eight native predicates retain their literal endpoint pairs and
100*factory-tech-eps tolerance. Displaced 3D boxes remain supplemental.

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
        assert not cj.isSubset([-0.7, -0.5, 1, 2], [-1, 1, -1, 1], TOL)

    def test_source_3d_domains(self):
        # Original pass(7)-(8), including rejection on the second axis.
        B = [-1, 1, -1, 1, -1, 1]
        assert cj.isSubset([-1, 1, 0, 1, -1, 1], B, TOL)
        assert not cj.isSubset([-1, 1, -2, 1, -1, 1], B, TOL)

    def test_supplemental_3d_domains(self):
        A = [0, 1, 0, 1, 0, 1]
        assert cj.isSubset(A, [-1, 2, -1, 2, -1, 2], TOL)
        assert not cj.isSubset(A, [-1, 2, -1, 2, 0.5, 2], TOL)
