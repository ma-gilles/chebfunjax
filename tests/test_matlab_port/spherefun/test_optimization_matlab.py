"""Port of MATLAB Chebfun tests/spherefun/test_optimization.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_optimization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no max2/min2")


class TestSpherefunOptimization:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
