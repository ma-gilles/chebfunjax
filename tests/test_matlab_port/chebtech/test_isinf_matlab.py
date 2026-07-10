"""Port of MATLAB Chebfun tests/chebtech/test_isinf.m (Opus 4.8).

chebfunjax Chebtech has NO ``isinf()`` method, so every assertion in this
file is skipped with a precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no isinf() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsinf:
    def test_scalar_inf_is_inf(self, Tech):
        # pass(n,1): isinf(make({[], y})) with y(4) = inf
        pytest.skip(_REASON)

    def test_array_inf_is_inf(self, Tech):
        # pass(n,2): isinf(make({[], y})) with y(4) = inf
        pytest.skip(_REASON)

    def test_finite_scalar_not_inf(self, Tech):
        # pass(n,3): ~isinf(make(@(x) x))
        pytest.skip(_REASON)

    def test_finite_array_not_inf(self, Tech):
        # pass(n,4): ~isinf(make(@(x) [x, x]))
        pytest.skip(_REASON)
