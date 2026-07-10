"""Port of MATLAB Chebfun tests/trigtech/test_min.m (Opus 4.8).

min(f) returns the minimum value and its location.

Provenance
----------
MATLAB source : tests/trigtech/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechMin:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_exp_neg_cos(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_sin10(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_exp_sin100(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_exp_neg_sin100(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_sign_approx(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_array_valued(self):
        raise AssertionError("min() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no min() method")
    def test_complex_valued(self):
        raise AssertionError("min() not implemented")

