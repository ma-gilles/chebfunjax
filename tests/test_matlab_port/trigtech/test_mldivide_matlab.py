"""Port of MATLAB Chebfun tests/trigtech/test_mldivide.m (Opus 4.8).

mldivide (\\) solves least-squares against a trigtech basis.

Provenance
----------
MATLAB source : tests/trigtech/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMldivide:
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_scalar_exact(self):
        raise AssertionError("mldivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_array_exact_coeff(self):
        raise AssertionError("mldivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_array_exact_residual(self):
        raise AssertionError("mldivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_least_squares(self):
        raise AssertionError("mldivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_error_nontrigtech(self):
        raise AssertionError("mldivide not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech / mldivide")
    def test_error_identifier(self):
        raise AssertionError("mldivide not implemented")

