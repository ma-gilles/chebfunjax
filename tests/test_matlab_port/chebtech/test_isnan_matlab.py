"""Port of MATLAB Chebfun tests/chebtech/test_isnan.m (Opus 4.8).

chebfunjax Chebtech has NO ``isnan()`` method, so every assertion in this
file is skipped with a precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no isnan() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsnan:
    def test_scalar_not_nan(self, Tech):
        # pass(n,1): ~isnan(make(@(x) x))
        pytest.skip(_REASON)

    def test_array_not_nan(self, Tech):
        # pass(n,2): ~isnan(make(@(x) [x, x.^2]))
        pytest.skip(_REASON)

    def test_constant_nan_is_nan(self, Tech):
        # pass(n,3): isnan(make(NaN))
        pytest.skip(_REASON)

    def test_scalar_nan_is_nan(self, Tech):
        # pass(n,4): isnan(make(@(x) x + NaN)) (or NaN/Inf construction error)
        pytest.skip(_REASON)

    def test_array_nan_is_nan(self, Tech):
        # pass(n,5): isnan(make(@(x) [x, x + NaN])) (or NaN/Inf construction error)
        pytest.skip(_REASON)
