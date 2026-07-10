"""Port of MATLAB Chebfun tests/trigtech/test_isnan.m (Opus 4.8).

isnan(f) is true if any coefficient/value is NaN.

Provenance
----------
MATLAB source : tests/trigtech/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechIsnan:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan() method")
    def test_finite_scalar(self):
        raise AssertionError("isnan() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan() method")
    def test_finite_array(self):
        raise AssertionError("isnan() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan() method")
    def test_nan_constant(self):
        raise AssertionError("isnan() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan() method")
    def test_nan_scalar(self):
        raise AssertionError("isnan() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isnan() method")
    def test_nan_array(self):
        raise AssertionError("isnan() not implemented")

