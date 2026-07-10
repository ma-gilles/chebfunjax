"""Port of MATLAB Chebfun tests/trigtech/test_minandmax.m (Opus 4.8).

minandmax(f) returns both extrema and their locations.

Provenance
----------
MATLAB source : tests/trigtech/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMinandmax:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_exp_neg_cos(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_sin10(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_exp_sin100(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_exp_neg_sin100(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_sign_approx(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_array_valued(self):
        raise AssertionError("minandmax() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no minandmax() method")
    def test_complex_array_valued(self):
        raise AssertionError("minandmax() not implemented")

