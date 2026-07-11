"""Port of MATLAB Chebfun tests/chebfun3/test_repeatedArithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_repeatedArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="repeated arithmetic needs simplify/compression on Chebfun3")


class TestChebfun3Repeatedarithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
