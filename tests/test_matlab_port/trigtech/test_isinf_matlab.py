"""Port of MATLAB Chebfun tests/trigtech/test_isinf.m (Opus 4.8).

isinf(f) is true if any value is Inf.

Provenance
----------
MATLAB source : tests/trigtech/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechIsinf:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no isinf() method")
    def test_scalar_inf(self):
        raise AssertionError("isinf() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isinf() method")
    def test_array_inf(self):
        raise AssertionError("isinf() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isinf() method")
    def test_finite_scalar(self):
        raise AssertionError("isinf() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no isinf() method")
    def test_finite_array(self):
        raise AssertionError("isinf() not implemented")

