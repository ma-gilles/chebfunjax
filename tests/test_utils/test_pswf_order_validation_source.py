"""PSWF source input assertion, pswf.m at Chebfun7574c77."""

import importlib

import numpy as np
import pytest

module = importlib.import_module("chebfunjax.utils.pswf")


@pytest.mark.parametrize("orders", [0.5, -0.5, np.array([[0, 1], [2, 3]])])
def test_source_orders_rejected_before_numerical_builder(orders, monkeypatch):
    def invalid_builder(*args):
        pytest.fail("invalid source orders reached numerical builder")

    monkeypatch.setattr(module, "_build_legendre_coeffs", invalid_builder)
    with pytest.raises(ValueError, match="non-negative integer"):
        module.pswf(orders, 1, output="chebfun")


@pytest.mark.parametrize("orders", [np.array([[0, 1]]), np.array([[0], [1]])])
def test_source_row_and_column_order_vectors_retained(orders):
    p, eigenvalues = module.pswf(orders, 1, output="chebfun")
    assert p.n_columns == 2
    assert np.asarray(eigenvalues).shape == (2,)
