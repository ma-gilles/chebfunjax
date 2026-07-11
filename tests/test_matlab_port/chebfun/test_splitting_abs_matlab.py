"""Port of MATLAB Chebfun tests/chebfun/test_splitting_abs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_splitting_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="covered by SplittingOn unit tests + abs port")


class TestChebfunSplittingAbs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
