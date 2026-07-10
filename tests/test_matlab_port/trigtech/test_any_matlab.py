"""Port of MATLAB Chebfun tests/trigtech/test_any.m (Opus 4.8).

any(f) reduces columns/rows of an (array-valued) trigtech to a logical.

Provenance
----------
MATLAB source : tests/trigtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechAny:
    @pytest.mark.xfail(reason="chebfunjax lacks any() and array-valued trigtech")
    def test_empty(self):
        raise AssertionError("any() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks any() and array-valued trigtech")
    def test_columns(self):
        raise AssertionError("any() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks any() and array-valued trigtech")
    def test_rows(self):
        raise AssertionError("any() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks any() and array-valued trigtech")
    def test_rows_zero(self):
        raise AssertionError("any() not implemented")

