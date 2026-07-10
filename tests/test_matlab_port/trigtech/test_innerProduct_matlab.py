"""Port of MATLAB Chebfun tests/trigtech/test_innerProduct.m (Opus 4.8).

innerProduct(f,g) = integral of conj(f).*g over [-1,1].

Provenance
----------
MATLAB source : tests/trigtech/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechInnerProduct:
    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_orthogonality_sin_cos(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_orthogonality_sin_cos4(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_known_result_exp(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_known_result_sin4(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_sesquilinearity(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_conjugate_symmetry(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_additivity_left(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_additivity_right(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_nonnegative_norm(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_array_valued(self):
        raise AssertionError("innerProduct() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no innerProduct() method")
    def test_error_nonchebtech(self):
        raise AssertionError("innerProduct() not implemented")

