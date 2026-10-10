"""Port of MATLAB Chebfun tests/chebtech/test_isreal.m (Opus 4.8).

The tests exercise the public isreal method with the native inputs.

Provenance
----------
MATLAB source : tests/chebtech/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsreal:
    def test_complex_is_not_real(self, Tech):
        # pass(n,1): ~isreal(make(@(x) sin(x) + 1i*cos(x)))
        f = Tech.from_function(lambda x: jnp.sin(x) + 1j * jnp.cos(x))
        assert not f.isreal()

    def test_imaginary_is_not_real(self, Tech):
        # pass(n,2): ~isreal(make(@(x) 1i*cos(x)))
        f = Tech.from_function(lambda x: 1j * jnp.cos(x))
        assert not f.isreal()

    def test_real_is_real(self, Tech):
        # pass(n,3): isreal(make(@(x) sin(x)))
        f = Tech.from_function(jnp.sin)
        assert f.isreal()

    # FIXED (Fable 5, Big-Three array-valued epic): array-valued isreal.
    def test_array_complex_first_col(self, Tech):
        # pass(n,4): ~isreal(make(@(x) [sin(x) + 1i*cos(x), exp(x)]))
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x) + 1j * jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        assert not f.isreal()

    def test_array_imaginary_first_col(self, Tech):
        # pass(n,5): ~isreal(make(@(x) [1i*cos(x), exp(x)]))
        f = Tech.from_function(
            lambda x: jnp.stack([1j * jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        assert not f.isreal()

    def test_array_all_real(self, Tech):
        # pass(n,6): isreal(make(@(x) [sin(x), exp(x)]))
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1)
        )
        assert f.isreal()
