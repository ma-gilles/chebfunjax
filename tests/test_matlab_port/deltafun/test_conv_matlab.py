"""Port of MATLAB Chebfun tests/deltafun/test_conv.m (Opus 4.8).

chebfunjax's Deltafun has no ``conv`` (convolution) method, so every assertion
in this MATLAB test is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_conv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no conv (convolution) method"
)


class TestDeltafunConv:
    def test_conv_empty(self):
        # pass(1): isempty(conv(d,d)) && isempty(conv(d,d1)) && isempty(conv(d1,d))
        pass

    def test_conv_delta_with_smooth(self):
        # pass(2): conv(d1, d2) recovers f = @(x) x within deltaTol
        pass

    def test_conv_delta_with_delta_prime(self):
        # pass(3): conv(d1, diff(d2)) recovers 1 within deltaTol
        pass
