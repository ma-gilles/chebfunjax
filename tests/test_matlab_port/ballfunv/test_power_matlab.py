"""Port of MATLAB Chebfun tests/ballfunv/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="ballfunv feature 'power' not implemented (MATLAB-specific accessor or missing op)")


class TestBallfunvPower:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
