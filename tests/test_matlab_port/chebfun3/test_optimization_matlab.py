"""Port of MATLAB Chebfun tests/chebfun3/test_optimization.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="requires max3/min3 (absent)")


class TestChebfun3Optimization:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
