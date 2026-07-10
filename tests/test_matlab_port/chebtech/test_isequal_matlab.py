"""Port of MATLAB Chebfun tests/chebtech/test_isequal.m (Opus 4.8).

chebfunjax Chebtech has NO ``isequal()`` method (there is no equality
predicate on techs), so every assertion in this file is skipped with a
precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no isequal() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsequal:
    def test_equal_to_self(self, Tech):
        # pass(n,1): isequal(f, g) && isequal(g, f) with g = f
        pytest.skip(_REASON)

    def test_not_equal_different_function(self, Tech):
        # pass(n,2): ~isequal(sin, cos)
        pytest.skip(_REASON)

    def test_not_equal_scalar_vs_array(self, Tech):
        # pass(n,3): ~isequal(sin, [sin cos])
        pytest.skip(_REASON)

    def test_equal_same_array(self, Tech):
        # pass(n,4): isequal(f, g) with f = g = [sin cos]
        pytest.skip(_REASON)

    def test_not_equal_different_arrays(self, Tech):
        # pass(n,5): ~isequal([sin cos], [sin exp])
        pytest.skip(_REASON)
