"""Port of MATLAB Chebfun tests/chebfun/test_doubleLength.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_doubleLength.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no doubleLength")


class TestChebfunDoublelength:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
