"""Port of MATLAB Chebfun tests/chebtech/test_isequal.m (Opus 4.8).

The tests exercise the public isequal method with the native inputs.

Provenance
----------
MATLAB source : tests/chebtech/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsequal:
    def test_equal_to_self(self, Tech):
        # pass(n,1): isequal(f, g) && isequal(g, f) with g = f
        f = Tech.from_function(jnp.sin)
        g = f
        assert f.isequal(g) and g.isequal(f)

    def test_not_equal_different_function(self, Tech):
        # pass(n,2): ~isequal(sin, cos)
        f = Tech.from_function(jnp.sin)
        g = Tech.from_function(jnp.cos)
        assert not f.isequal(g)

    def test_not_equal_scalar_vs_array(self, Tech):
        # pass(n,3): ~isequal(sin, [sin cos])
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued (n, m)
        # techs exist; a scalar (n,) and an array (n, 2) differ in shape.
        f = Tech.from_function(jnp.sin)
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        assert not f.isequal(g)

    def test_equal_same_array(self, Tech):
        # pass(n,4): isequal(f, g) with f = g = [sin cos]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = f
        assert f.isequal(g)

    def test_not_equal_different_arrays(self, Tech):
        # pass(n,5): ~isequal([sin cos], [sin exp])
        # FIXED (Fable 5, Big-Three array-valued epic): equal shapes but the
        # second column differs => not equal.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1)
        )
        assert not f.isequal(g)
