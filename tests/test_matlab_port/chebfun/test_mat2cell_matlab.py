"""Port of MATLAB Chebfun tests/chebfun/test_mat2cell.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no array-valued (multi-column) chebfun")


class TestChebfunMat2cell:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
