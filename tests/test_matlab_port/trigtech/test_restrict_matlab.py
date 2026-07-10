"""Port of MATLAB Chebfun tests/trigtech/test_restrict.m (Opus 4.8).

restrict(f, subint) restricts f to a subinterval (returns a bndfun).

Provenance
----------
MATLAB source : tests/trigtech/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechRestrict:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_empty(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_whole_interval(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_error_right(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_error_left(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_error_nonmonotone(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_spotcheck_sin(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_spotcheck_rational(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_spotcheck_cos(self):
        raise AssertionError("restrict() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no restrict() method")
    def test_multiple_subintervals(self):
        raise AssertionError("restrict() not implemented")

