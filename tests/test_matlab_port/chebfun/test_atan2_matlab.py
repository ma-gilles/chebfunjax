"""Port of MATLAB Chebfun tests/chebfun/test_atan2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_atan2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no atan2")


class TestChebfunAtan2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
