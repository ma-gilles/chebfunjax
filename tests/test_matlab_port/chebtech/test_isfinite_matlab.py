"""Port of MATLAB Chebfun tests/chebtech/test_isfinite.m (Opus 4.8).

chebfunjax Chebtech has NO ``isfinite()`` method, so every assertion in
this file is skipped with a precise reason.  No assertion is silently
dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no isfinite() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsfinite:
    def test_scalar_inf_not_finite(self, Tech):
        # pass(n,1): ~isfinite(make({[], y})) with y(4) = inf
        pytest.skip(_REASON)

    def test_array_inf_not_finite(self, Tech):
        # pass(n,2): ~isfinite(make({[], y})) with y(4) = inf
        pytest.skip(_REASON)

    def test_finite_scalar_is_finite(self, Tech):
        # pass(n,3): isfinite(make(@(x) x))
        pytest.skip(_REASON)

    def test_finite_array_is_finite(self, Tech):
        # pass(n,4): isfinite(make(@(x) [x, x]))
        pytest.skip(_REASON)
