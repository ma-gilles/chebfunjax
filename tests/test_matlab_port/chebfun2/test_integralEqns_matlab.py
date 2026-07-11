"""Port of MATLAB Chebfun tests/chebfun2/test_integralEqns.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_integralEqns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Fredholm integral-equation solves need fred/volt on Chebfun2 (absent)")


class TestChebfun2Integraleqns:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
