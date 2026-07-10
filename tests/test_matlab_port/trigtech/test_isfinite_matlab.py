"""Port of MATLAB Chebfun tests/trigtech/test_isfinite.m (Opus 4.8).

isfinite(f) is false if any coefficient/value is Inf.

Provenance
----------
MATLAB source : tests/trigtech/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechIsfinite:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no isfinite() method")
    def test_scalar_inf(self):
        raise AssertionError("isfinite() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isfinite() method")
    def test_array_inf(self):
        raise AssertionError("isfinite() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isfinite() method")
    def test_finite_scalar(self):
        raise AssertionError("isfinite() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isfinite() method")
    def test_finite_array(self):
        raise AssertionError("isfinite() not implemented")

