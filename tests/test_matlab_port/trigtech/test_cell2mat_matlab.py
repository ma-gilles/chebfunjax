"""Port of MATLAB Chebfun tests/trigtech/test_cell2mat.m (Opus 4.8).

cell2mat concatenates trigtechs into an array-valued trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_cell2mat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechCell2mat:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / cell2mat")
    def test_concatenate(self):
        raise AssertionError("cell2mat not implemented")

