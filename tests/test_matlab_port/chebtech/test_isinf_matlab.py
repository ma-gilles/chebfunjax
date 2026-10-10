"""Port of MATLAB Chebfun tests/chebtech/test_isinf.m (Fable 5).

The tests exercise the public isinf method with the native inputs.

Provenance
----------
MATLAB source : tests/chebtech/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsinf:
    def test_scalar_inf_is_inf(self, Tech):
        # pass(n,1): isinf(make({[], y})) with y(4) = inf
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_coeffs(y)
        assert f.isinf()

    def test_array_inf_is_inf(self, Tech):
        # pass(n,2): native source repeats the same scalar coefficient fixture.
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_coeffs(y)
        assert f.isinf()

    def test_finite_scalar_not_inf(self, Tech):
        # pass(n,3): ~isinf(make(@(x) x))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: x)
        assert not f.isinf()

    def test_finite_array_not_inf(self, Tech):
        # pass(n,4): ~isinf(make(@(x) [x, x]))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([x, x], axis=-1))
        assert not f.isinf()


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_isinf_supplemental_array(Tech):
    y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
    f = Tech.from_coeffs(jnp.stack([y, jnp.ones_like(y)], axis=-1))
    assert f.isinf()
