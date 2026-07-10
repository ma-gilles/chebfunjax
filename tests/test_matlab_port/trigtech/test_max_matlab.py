"""Port of MATLAB Chebfun tests/trigtech/test_max.m (Opus 4.8).

max(f) returns the maximum value and its location.

Provenance
----------
MATLAB source : tests/trigtech/test_max.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMax:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_exp_neg_cos(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_sin10(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_exp_sin100(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_exp_neg_sin100(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_sign_approx(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_array_valued(self):
        raise AssertionError("max() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no max() method")
    def test_complex_valued(self):
        raise AssertionError("max() not implemented")

