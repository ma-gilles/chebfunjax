"""Port of MATLAB Chebfun tests/diskfun/test_sum.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="sum(dim) partial integrals not implemented (sum2 total exists as .sum())")


class TestDiskfunSum:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
