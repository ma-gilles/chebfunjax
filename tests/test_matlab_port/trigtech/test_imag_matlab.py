"""Port of MATLAB Chebfun tests/trigtech/test_imag.m (Opus 4.8).

imag(f) extracts the imaginary part of a trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechImag:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no imag() method")
    def test_scalar(self):
        raise AssertionError("imag() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no imag() method")
    def test_array(self):
        raise AssertionError("imag() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no imag() method")
    def test_real_function(self):
        raise AssertionError("imag() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no imag() method")
    def test_real_array(self):
        raise AssertionError("imag() not implemented")

