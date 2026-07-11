"""Port of MATLAB Chebfun tests/diskfun/test_optimization.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no max2/min2")


class TestDiskfunOptimization:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
