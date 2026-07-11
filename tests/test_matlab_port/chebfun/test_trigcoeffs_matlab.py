"""Port of MATLAB Chebfun tests/chebfun/test_trigcoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no trigcoeffs accessor (trig tech coeffs internal)")


class TestChebfunTrigcoeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
