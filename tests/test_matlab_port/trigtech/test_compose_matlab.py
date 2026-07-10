"""Port of MATLAB Chebfun tests/trigtech/test_compose.m (Opus 4.8).

compose forms g(f) / binary compositions, re-resolving adaptively.

Provenance
----------
MATLAB source : tests/trigtech/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechCompose:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_scalar_sin(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_array_sin(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_array_sin_values(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_array3_sin_values(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_binary_plus(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_binary_times_array(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_g_of_f(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_g_of_f_array_g(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_g_of_f_array_f(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_error_array_array(self):
        raise AssertionError("compose() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no compose() method")
    def test_error_dim(self):
        raise AssertionError("compose() not implemented")

