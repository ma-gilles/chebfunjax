"""Port of MATLAB Chebfun tests/chebtech/test_imag.m (Opus 4.8).

chebfunjax Chebtech has NO ``imag()`` method, so every assertion in this
file is skipped with a precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no imag() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechImag:
    def test_imag_scalar(self, Tech):
        # pass(n,1): imag(exp(x) + 1i*sin(x)) == sin(x)
        pytest.skip(_REASON)

    def test_imag_array(self, Tech):
        # pass(n,2): imag([exp(x)+1i*sin(x), -exp(1i*x)]) columns
        pytest.skip(_REASON)

    def test_imag_of_real_is_zero(self, Tech):
        # pass(n,3): imag(cos(x)) has a single coeff
        pytest.skip(_REASON)

    def test_imag_array_of_real_is_zero(self, Tech):
        # pass(n,4): imag([cos sin exp]) is [1,3] zeros
        pytest.skip(_REASON)
