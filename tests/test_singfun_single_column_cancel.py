"""Supplemental singleton-storage controls, separate from native cumsum30."""

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


def test_scalar_and_single_column_endpoint_cancellation():
    for column in (False, True):
        coefficients = jnp.array([1.0, 1.0])
        if column:
            coefficients = coefficients[:, None]
        f = Singfun(Chebtech2.from_coeffs(coefficients), (-1.0, 0.0))
        g = f.cancelExponents()
        assert g.exponents == (0.0, 0.0)
        np.testing.assert_array_equal(
            np.asarray(g(jnp.array([-0.75, 0.0, 0.75]))).reshape(-1), np.ones(3)
        )
        positive = Singfun(Chebtech2.from_coeffs(coefficients), (0.5, 0.0))
        assert positive.cancelExponents() is positive
        constant = coefficients[:1]
        no_root = Singfun(Chebtech2.from_coeffs(constant), (-1.0, 0.0))
        assert no_root.cancelExponents() is no_root


def test_scalar_and_single_column_cumsum_storage():
    scalar = Singfun(Chebtech2.from_coeffs(jnp.array([1.0, 0.25])), (0.5, 0.0))
    column = Singfun(Chebtech2.from_coeffs(jnp.array([[1.0], [0.25]])), (0.5, 0.0))
    a, b = scalar.cumsum(), column.cumsum()
    assert b.smoothPart.coeffs.shape == (a.smoothPart.coeffs.shape[0], 1)
    np.testing.assert_array_equal(
        np.asarray(b.smoothPart.coeffs[:, 0]), np.asarray(a.smoothPart.coeffs)
    )
    x = jnp.array([-0.75, 0.0, 0.75])
    assert b(x).shape == x.shape  # Existing scalar Singfun evaluation adapter.
    np.testing.assert_array_equal(np.asarray(b(x)), np.asarray(a(x)))
