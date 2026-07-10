"""Port of MATLAB Chebfun tests/trigtech/test_circconv.m (Opus 4.8).

circconv is periodic (circular) convolution of two trigtechs.

Provenance
----------
MATLAB source : tests/trigtech/test_circconv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechCircconv:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no circconv() method")
    def test_empty(self):
        raise AssertionError("circconv() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no circconv() method")
    def test_odd_functions_zero(self):
        raise AssertionError("circconv() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no circconv() method")
    def test_self_convolution_at_zero(self):
        raise AssertionError("circconv() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no circconv() method")
    def test_self_convolution_at_x(self):
        raise AssertionError("circconv() not implemented")

