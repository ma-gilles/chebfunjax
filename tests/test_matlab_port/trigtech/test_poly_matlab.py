"""Port of MATLAB Chebfun tests/trigtech/test_poly.m (Opus 4.8).

poly(f) returns the Fourier coefficients as a polynomial-style row.

Provenance
----------
MATLAB source : tests/trigtech/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechPoly:
    @pytest.mark.xfail(reason="chebfunjax exposes coeffs directly but not the poly() row layout (array-valued)")
    def test_zeros(self):
        raise AssertionError("poly() not implemented")

    @pytest.mark.xfail(reason="chebfunjax exposes coeffs directly but not the poly() row layout (array-valued)")
    def test_constant(self):
        raise AssertionError("poly() not implemented")

    @pytest.mark.xfail(reason="chebfunjax exposes coeffs directly but not the poly() row layout (array-valued)")
    def test_one_plus_cos(self):
        raise AssertionError("poly() not implemented")

    @pytest.mark.xfail(reason="chebfunjax exposes coeffs directly but not the poly() row layout (array-valued)")
    def test_complex_exponentials(self):
        raise AssertionError("poly() not implemented")

    @pytest.mark.xfail(reason="chebfunjax exposes coeffs directly but not the poly() row layout (array-valued)")
    def test_array_valued(self):
        raise AssertionError("poly() not implemented")

