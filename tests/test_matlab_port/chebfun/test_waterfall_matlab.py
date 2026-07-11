"""Port of MATLAB Chebfun tests/chebfun/test_waterfall.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_waterfall.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="plot smoke test")


class TestChebfunWaterfall:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
