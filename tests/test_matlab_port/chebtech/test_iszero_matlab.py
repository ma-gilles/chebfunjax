"""Port of MATLAB Chebfun tests/chebtech/test_iszero.m (Fable 5).

These tests exercise the public per-column iszero method with the exact
native coefficient arrays, including NaNs and row/column distinctions.

Provenance
----------
MATLAB source : tests/chebtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIszero:
    def test_iszero_columns_mixed(self, Tech):
        # pass(n,1): iszero(f) == [1 0 0] for coeffs [0 1 0; 0 0 NaN]
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs supported.
        f = Tech.from_coeffs(
            jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, jnp.nan]], dtype=jnp.float64)
        )
        assert list(np.asarray(f.iszero()).astype(int)) == [1, 0, 0]

    def test_iszero_row_mixed(self, Tech):
        # pass(n,2): iszero(f) == [1 0 0] for coeffs [0 NaN 1]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([[0.0, jnp.nan, 1.0]], dtype=jnp.float64))
        assert list(np.asarray(f.iszero()).astype(int)) == [1, 0, 0]

    def test_iszero_column_mixed(self, Tech):
        # pass(n,3): iszero(f) == 0 for coeffs [0 NaN 1]'
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([0.0, jnp.nan, 1.0], dtype=jnp.float64))
        assert int(f.iszero()) == 0

    def test_iszero_all_zero(self, Tech):
        # pass(n,4): iszero(f) == 1 for coeffs zeros(3,1)
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.zeros(3, dtype=jnp.float64))
        assert int(f.iszero()) == 1

    def test_iszero_nan(self, Tech):
        # pass(n,5): iszero(f) == 0 for coeffs NaN
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([jnp.nan], dtype=jnp.float64))
        assert int(f.iszero()) == 0


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_iszero_jit_complex_and_empty(Tech):
    # Supplemental API controls: exact zeros, complex/nonfinite columns,
    # tracing, native empty 0-by-0 and explicit zero-row column storage.
    coefficients = jnp.asarray([[0, 1j, jnp.inf, jnp.nan], [0, 0, 0, 0]])
    result = jax.jit(lambda c: Tech.from_coeffs(c).iszero())(coefficients)
    assert np.array_equal(result, [True, False, False, False])
    assert Tech.empty().iszero().shape == (0,)
    assert Tech.from_coeffs(jnp.asarray([])).iszero().shape == (0,)
    assert np.array_equal(Tech(coeffs=jnp.empty((0, 3))).iszero(), [True]*3)
