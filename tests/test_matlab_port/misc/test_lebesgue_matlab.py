"""Port of MATLAB Chebfun tests/misc/test_lebesgue.m (Fable 5).

MATLAB lebesgue(x, dom) -> (L, C); chebfunjax exposes
lebesgue_constant / lebesgue_function.

Provenance
----------
MATLAB source : tests/misc/test_lebesgue.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.lebesgue import lebesgue_constant
from chebfunjax.utils.quadrature import chebpts, legpts

EPS = float(np.finfo(np.float64).eps)


class TestLebesgue:
    def test_three_chebyshev_points(self):
        C = float(lebesgue_constant(np.asarray(chebpts(3))))
        assert abs(C - 5 / 4) < 10 * EPS * 10

    def test_three_legendre_points(self):
        x, _ = legpts(3)
        C = float(lebesgue_constant(np.asarray(x)))
        assert abs(C - 7 / 3) < 1e-8

    def test_equispaced_shifted(self):
        C = float(lebesgue_constant(np.linspace(5, 9, 3),
                                    domain=(5.0, 9.0)))
        assert abs(C - 5 / 4) < 1e-10

    def test_two_points_wide_interval(self):
        # MATLAB: L = lebesgue([1 2], [0 7]); norm(L, inf) == 11
        C = float(lebesgue_constant(np.array([1.0, 2.0]),
                                    domain=(0.0, 7.0)))
        assert abs(C - 11.0) < 1e-8

    def test_three_points_asymmetric(self):
        C = float(lebesgue_constant(np.array([-1.0, 0.0, 1.0]),
                                    domain=(-1.0, 2.0)))
        assert abs(C - 7.0) < 1e-8
