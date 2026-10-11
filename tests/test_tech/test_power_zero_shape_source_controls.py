"""Supplemental exact shape controls; no native techPOWER test exists."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("columns", [None, 1, 3])
@pytest.mark.parametrize("complex_data", [False, True])
def test_zero_power_ones_shape(Tech, columns, complex_data):
    shape = (2,) if columns is None else (2, columns)
    a = np.full(shape, 2.0, dtype=complex if complex_data else float)
    a[1] = 0.25 + 0.125j if complex_data else 0.25
    f = Tech.from_coeffs(jnp.asarray(a))
    h = f**0
    expected = (1,) if columns is None else (1, columns)
    assert type(h) is Tech and h.coeffs.shape == expected
    np.testing.assert_array_equal(np.asarray(h.coeffs), np.ones(expected))
    np.testing.assert_array_equal(np.asarray(f.coeffs), a)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_empty_zero_power_native_semantics(Tech):
    # Native tpow1 coeff storage(17,0); Python empty sentinel is an adapter.
    f = Tech.empty()
    h = f**0
    assert type(h) is Tech and h.isempty()
