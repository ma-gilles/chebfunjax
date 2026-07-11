"""Port of MATLAB Chebfun tests/diskfun/test_partitionCombine.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_partitionCombine.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="partition internal")


class TestDiskfunPartitioncombine:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
