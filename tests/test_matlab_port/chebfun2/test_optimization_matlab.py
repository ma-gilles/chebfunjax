"""Port of MATLAB Chebfun tests/chebfun2/test_optimization.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="requires minandmax2/max2/min2 (absent)")


class TestChebfun2Optimization:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
