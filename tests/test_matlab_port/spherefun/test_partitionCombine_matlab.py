"""Port of MATLAB Chebfun tests/spherefun/test_partitionCombine.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_partitionCombine.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun partition/combine not exposed")


class TestSpherefunPartitioncombine:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
