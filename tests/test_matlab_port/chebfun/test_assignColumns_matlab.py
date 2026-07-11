"""Port of MATLAB Chebfun tests/chebfun/test_assignColumns.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no array-valued (multi-column) chebfun")


class TestChebfunAssigncolumns:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
