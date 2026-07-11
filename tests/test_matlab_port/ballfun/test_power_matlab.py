"""Port of MATLAB Chebfun tests/ballfun/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no power composition")


class TestBallfunPower:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
