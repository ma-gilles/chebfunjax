"""Port of MATLAB Chebfun tests/classicfun/test_mat2cell.m (Fable 5).

Provenance
----------
MATLAB source : tests/classicfun/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no array-valued Bndfun to split with mat2cell")


class TestClassicfunMat2cell:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
