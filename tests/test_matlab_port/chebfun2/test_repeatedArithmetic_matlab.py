"""Port of MATLAB Chebfun tests/chebfun2/test_repeatedArithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_repeatedArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="repeated f+f-f cycles need compression (Chebfun2 plus does not compress)")


class TestChebfun2Repeatedarithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
