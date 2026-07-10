"""Port of MATLAB Chebfun tests/trigtech/test_flipud.m (Opus 4.8).

flipud(f) maps f(x) -> f(-x) (reverses the domain).

Provenance
----------
MATLAB source : tests/trigtech/test_flipud.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechFlipud:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no flipud() method")
    def test_scalar(self):
        raise AssertionError("flipud() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no flipud() method")
    def test_array(self):
        raise AssertionError("flipud() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no flipud() method")
    def test_even_length(self):
        raise AssertionError("flipud() not implemented")

