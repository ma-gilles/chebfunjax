"""Port of MATLAB Chebfun tests/chebfun/test_find.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_find.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no find (logical chebfun indexing)")


class TestChebfunFind:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
