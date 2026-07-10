"""Port of MATLAB Chebfun tests/trigtech/test_real.m (Opus 4.8).

real(f) extracts the real part of a trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechReal:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no real() method")
    def test_scalar(self):
        raise AssertionError("real() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no real() method")
    def test_array(self):
        raise AssertionError("real() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no real() method")
    def test_imaginary_function(self):
        raise AssertionError("real() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no real() method")
    def test_imaginary_array(self):
        raise AssertionError("real() not implemented")

