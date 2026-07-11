"""Port of MATLAB Chebfun tests/chebfun/test_fix.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_fix.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no fix (floor/ceil/round ported)")


class TestChebfunFix:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
