"""Port of MATLAB Chebfun tests/deltafun/test_innerProduct.m (Opus 4.8).

chebfunjax's Deltafun has no ``innerProduct`` method (the distribution/test-
function pairing that evaluates the smooth part and its derivatives at each
delta location), so every assertion in this MATLAB test is skipped with a
precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no innerProduct() method"
)


class TestDeltafunInnerProduct:
    def test_innerproduct_empty(self):
        # pass(1): innerProduct with an empty deltafun is empty
        pass

    def test_innerproduct_delta_with_smooth(self):
        # pass(2): <g,f> = sum(mag.*f(loc)) + <f,f>
        pass

    def test_innerproduct_deltaprime(self):
        # pass(3): <delta', g> = -g'(0)
        pass

    def test_innerproduct_diff_deltaprime(self):
        # pass(4): <diff(d), g> = g''(0)
        pass

    def test_innerproduct_higher_order_block(self):
        # pass(5): multi-order delta magnitude block pairing
        pass

    def test_innerproduct_self_positive_inf(self):
        # pass(6): <d,d> is +Inf
        pass

    def test_innerproduct_self_negative_inf(self):
        # pass(7): <d,-d> is -Inf
        pass
