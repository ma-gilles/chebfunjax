"""Port of MATLAB Chebfun tests/trigtech/test_assignColumns.m (Opus 4.8).

assignColumns writes columns of an array-valued trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechAssignColumns:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / assignColumns")
    def test_assign_two_columns(self):
        raise AssertionError("assignColumns not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / assignColumns")
    def test_assign_unhappy(self):
        raise AssertionError("assignColumns not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / assignColumns")
    def test_assign_empty_removes(self):
        raise AssertionError("assignColumns not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / assignColumns")
    def test_assign_scalar_column(self):
        raise AssertionError("assignColumns not implemented")

