"""Port of MATLAB Chebfun tests/trigtech/test_mrdivide.m (Opus 4.8).

mrdivide (/) divides a trigtech by a numeric matrix.

Provenance
----------
MATLAB source : tests/trigtech/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMrdivide:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_div_by_zero_nan(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_div_by_scalar(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_least_squares_identity(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_least_squares_row(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_scalar_over_function(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_row_over_array(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_error_dim(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_error_trigtech_div_trigtech(self):
        raise AssertionError("mrdivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mrdivide")
    def test_error_bad_arg(self):
        raise AssertionError("mrdivide not implemented")

