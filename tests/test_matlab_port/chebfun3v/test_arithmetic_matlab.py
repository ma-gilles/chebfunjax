"""Port of MATLAB Chebfun tests/chebfun3v/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'arithmetic' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vArithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
