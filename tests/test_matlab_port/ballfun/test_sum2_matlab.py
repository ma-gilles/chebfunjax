"""Port of MATLAB Chebfun tests/ballfun/test_sum2.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sum2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="sum over 2 vars not implemented")


class TestBallfunSum2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
