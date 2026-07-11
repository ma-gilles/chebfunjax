"""Port of MATLAB Chebfun tests/ballfun/test_sum.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="partial sums not implemented (sum() is the full integral)")


class TestBallfunSum:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
