"""Port of MATLAB Chebfun tests/chebfun/test_join.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_join.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no join")


class TestChebfunJoin:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
