"""Port of MATLAB Chebfun tests/chebtech/test_extractBoundaryRoots.m (Opus 4.8).

The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}`` and
factors out boundary roots (zeros at x = +/-1) of a chebtech via
``extractBoundaryRoots``.  chebfunjax implements NO ``extractBoundaryRoots``
on either Chebtech1 or Chebtech2, so every assertion is an honest xfail.

Provenance
----------
MATLAB source : tests/chebtech/test_extractBoundaryRoots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax lacks extractBoundaryRoots"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechExtractBoundaryRoots:
    def test_left_endpoint_roots(self, Tech):
        # pass(n, 1): sin(2x)*(1+x)^3 -> multiplicity l == 3 at x = -1.
        pytest.xfail(_REASON)

    def test_right_endpoint_roots(self, Tech):
        # pass(n, 2): sin(cos(3x))*(1-x)^2 -> multiplicity r == 2 at x = 1.
        pytest.xfail(_REASON)

    def test_both_endpoint_roots(self, Tech):
        # pass(n, 3): exp(x)*(1+x)*(1-x)^2 -> l == 1, r == 2.
        pytest.xfail(_REASON)

    def test_complex_case(self, Tech):
        # pass(n, 4): complex integrand with boundary roots.
        pytest.xfail(_REASON)

    def test_no_roots(self, Tech):
        # pass(n, 5): sin(1-x)/(1-x) has no boundary roots (l == r == 0).
        pytest.xfail(_REASON)

    def test_roots_not_explicit(self, Tech):
        # pass(n, 6): sin(1-x) has an implicit root at x = 1 (r == 1).
        pytest.xfail(_REASON)

    def test_array_valued(self, Tech):
        # pass(n, 7): array-valued boundary root extraction.
        pytest.xfail(_REASON)

    def test_full_arguments(self, Tech):
        # pass(n, 8): extractBoundaryRoots(f, [ml; mr]) with supplied mults.
        pytest.xfail(_REASON)

    def test_wrong_multiplicities(self, Tech):
        # pass(n, 9): supplied multiplicities exceed the true ones.
        pytest.xfail(_REASON)
