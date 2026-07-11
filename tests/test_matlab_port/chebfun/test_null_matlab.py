"""Port of MATLAB Chebfun tests/chebfun/test_null.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_null.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no array-valued (multi-column) chebfun")


class TestChebfunNull:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
