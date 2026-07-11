"""Port of MATLAB Chebfun tests/chebfun/test_bvp5c.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_bvp5c.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no bvp5c wrapper")


class TestChebfunBvp5c:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
