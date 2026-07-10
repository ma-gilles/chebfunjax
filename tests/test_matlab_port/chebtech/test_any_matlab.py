"""Port of MATLAB Chebfun tests/chebtech/test_any.m (Opus 4.8).

chebfunjax Chebtech has NO ``any()`` method (and is scalar-valued, so the
down-columns / across-rows semantics have no analogue).  Every assertion
in this file is skipped with a precise reason.  No assertion is silently
dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no any() method (and is scalar-valued)"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechAny:
    def test_any_empty_class(self, Tech):
        # pass(n,1): ~any(testclass)
        pytest.skip(_REASON)

    def test_any_down_columns(self, Tech):
        # pass(n,2): any(make(@(x) [sin(x) 0*x cos(x)])) == [1 0 1]
        pytest.skip(_REASON)

    def test_any_across_rows_nonzero(self, Tech):
        # pass(n,3): any(f, 2).coeffs == 1
        pytest.skip(_REASON)

    def test_any_across_rows_zero(self, Tech):
        # pass(n,4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        pytest.skip(_REASON)
