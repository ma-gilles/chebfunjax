"""Port of MATLAB Chebfun tests/trigtech/test_mat2cell.m (Opus 4.8).

mat2cell splits an array-valued trigtech into a cell of trigtechs.

Provenance
----------
MATLAB source : tests/trigtech/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMat2cell:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mat2cell")
    def test_split_first(self):
        raise AssertionError("mat2cell not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mat2cell")
    def test_split_rest(self):
        raise AssertionError("mat2cell not implemented")

