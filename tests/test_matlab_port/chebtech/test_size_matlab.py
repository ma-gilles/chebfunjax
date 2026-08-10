"""Port of MATLAB Chebfun tests/chebtech/test_size.m (Fable 5).

All three MATLAB assertions are ported, including the array-valued
(3-column) ones: chebfunjax techs store ``(n, m)`` coefficient matrices.
chebfunjax exposes no ``size()`` method returning ``[n, ncols]``; MATLAB's
``size(f) == size(f.coeffs)`` is asserted directly against
``f.coeffs.shape`` (with ``f.n``/``len(f)`` as the row count), and
``pref.fixedLength = 101`` maps to the ``n=101`` constructor argument.

Provenance
----------
MATLAB source : tests/chebtech/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechSize:
    def test_size_matches_coeffs(self, Tech):
        # pass(n,1): all(size(f) == size(f.coeffs)) for scalar sin
        f = Tech.from_function(jnp.sin)
        assert f.n == f.coeffs.shape[0]
        assert len(f) == f.coeffs.shape[0]

    def test_size_array_valued(self, Tech):
        # pass(n,2): all(size(f) == size(f.coeffs)) for [sin cos 1i*exp]
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1))
        assert f.n == f.coeffs.shape[0]
        assert f.coeffs.ndim == 2 and f.coeffs.shape[1] == 3

    def test_size_fixed_length_101(self, Tech):
        # pass(n,3): fixedLength=101 -> size(f) == [101, 3].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1),
            n=101)
        assert f.coeffs.shape == (101, 3)
