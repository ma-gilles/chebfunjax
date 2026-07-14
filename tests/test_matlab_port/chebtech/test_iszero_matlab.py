"""Port of MATLAB Chebfun tests/chebtech/test_iszero.m (Fable 5).

chebfunjax has no ``iszero()`` method, but MATLAB ``@chebtech/iszero.m`` is
exactly a per-column reduction over ``f.coeffs``::

    out = ~any(f.coeffs, 1);
    out = out & ~any(isnan(f.coeffs), 1);   % a NaN column is not zero

The faithful chebfunjax equivalent on an array-valued tech is
``all(coeffs == 0, axis=0) & ~any(isnan(coeffs), axis=0)``.  These tests
construct genuine (n, m) techs via ``from_coeffs`` (exercising array-valued
construction) and assert the MATLAB per-column result.

Provenance
----------
MATLAB source : tests/chebtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _iszero(f):
    """MATLAB @chebtech/iszero.m equivalent: per-column ``all zero, no NaN``."""
    zero = jnp.all(f.coeffs == 0, axis=0)
    return zero & ~jnp.any(jnp.isnan(f.coeffs), axis=0)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIszero:
    def test_iszero_columns_mixed(self, Tech):
        # pass(n,1): iszero(f) == [1 0 0] for coeffs [0 1 0; 0 0 NaN]
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs supported.
        f = Tech.from_coeffs(
            jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, jnp.nan]], dtype=jnp.float64)
        )
        assert list(np.asarray(_iszero(f)).astype(int)) == [1, 0, 0]

    def test_iszero_row_mixed(self, Tech):
        # pass(n,2): iszero(f) == [1 0 0] for coeffs [0 NaN 1]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([[0.0, jnp.nan, 1.0]], dtype=jnp.float64))
        assert list(np.asarray(_iszero(f)).astype(int)) == [1, 0, 0]

    def test_iszero_column_mixed(self, Tech):
        # pass(n,3): iszero(f) == 0 for coeffs [0 NaN 1]'
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([0.0, jnp.nan, 1.0], dtype=jnp.float64))
        assert int(_iszero(f)) == 0

    def test_iszero_all_zero(self, Tech):
        # pass(n,4): iszero(f) == 1 for coeffs zeros(3,1)
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.zeros(3, dtype=jnp.float64))
        assert int(_iszero(f)) == 1

    def test_iszero_nan(self, Tech):
        # pass(n,5): iszero(f) == 0 for coeffs NaN
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([jnp.nan], dtype=jnp.float64))
        assert int(_iszero(f)) == 0
