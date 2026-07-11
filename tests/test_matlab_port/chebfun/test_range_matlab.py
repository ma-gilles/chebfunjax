"""Port of MATLAB Chebfun tests/chebfun/test_range.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_range.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no range")


class TestChebfunRange:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
