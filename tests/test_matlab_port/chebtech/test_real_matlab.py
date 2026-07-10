"""Port of MATLAB Chebfun tests/chebtech/test_real.m (Opus 4.8).

chebfunjax Chebtech has NO ``real()`` method, so every assertion in this
file is skipped with a precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no real() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechReal:
    def test_real_scalar(self, Tech):
        # pass(n,1): real(exp(1i*x) + 1i*sin(x)) == cos(x)
        pytest.skip(_REASON)

    def test_real_array(self, Tech):
        # pass(n,2): real([exp(1i*x)+1i*sin(x), -exp(1i*x)]) columns
        pytest.skip(_REASON)

    def test_real_of_imaginary_is_zero(self, Tech):
        # pass(n,3): real(1i*cos(x)) has a single zero coeff
        pytest.skip(_REASON)

    def test_real_array_of_imaginary_is_zero(self, Tech):
        # pass(n,4): real(1i*[cos sin exp]) is [1,3] zeros
        pytest.skip(_REASON)
