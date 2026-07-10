"""Port of MATLAB Chebfun tests/trigtech/test_sign.m (Opus 4.8).

sign(f) is not implemented in chebfunjax's trigtech (there is no ``sign``
method), so every assertion is marked xfail with that precise reason.

Provenance
----------
MATLAB source : tests/trigtech/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechSign:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no sign() method")
    def test_positive_function(self):
        raise AssertionError("sign() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no sign() method")
    def test_negative_function(self):
        raise AssertionError("sign() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no sign() method")
    def test_complex_valued_function(self):
        raise AssertionError("sign() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no sign() method (and no array-valued trigtech)")
    def test_complex_array_valued(self):
        raise AssertionError("sign() not implemented")
