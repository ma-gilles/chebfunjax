"""Port of MATLAB Chebfun tests/trigtech/test_iszero.m (Opus 4.8).

iszero(f) reports which columns are identically zero.

Provenance
----------
MATLAB source : tests/trigtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechIszero:
    @pytest.mark.xfail(reason="chebfunjax lacks iszero() and mutable/array-valued values")
    def test_mixed_columns(self):
        raise AssertionError("iszero() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks iszero() and mutable/array-valued values")
    def test_row_vector_values(self):
        raise AssertionError("iszero() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks iszero() and mutable/array-valued values")
    def test_col_vector_values(self):
        raise AssertionError("iszero() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks iszero() and mutable/array-valued values")
    def test_all_zero(self):
        raise AssertionError("iszero() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks iszero() and mutable/array-valued values")
    def test_nan(self):
        raise AssertionError("iszero() not implemented")

