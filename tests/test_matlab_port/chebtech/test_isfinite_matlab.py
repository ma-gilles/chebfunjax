"""Port of MATLAB Chebfun tests/chebtech/test_isfinite.m (Fable 5).

The tests exercise the public isfinite method with the native inputs.

Provenance
----------
MATLAB source : tests/chebtech/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsfinite:
    def test_scalar_inf_not_finite(self, Tech):
        # pass(n,1): ~isfinite(make({[], y})) with y(4) = inf
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_coeffs(y)
        assert not f.isfinite()

    def test_array_inf_not_finite(self, Tech):
        # pass(n,2): native source repeats the same scalar coefficient fixture.
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_coeffs(y)
        assert not f.isfinite()

    def test_finite_scalar_is_finite(self, Tech):
        # pass(n,3): isfinite(make(@(x) x))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: x)
        assert f.isfinite()

    def test_finite_array_is_finite(self, Tech):
        # pass(n,4): isfinite(make(@(x) [x, x]))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([x, x], axis=-1))
        assert f.isfinite()


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_isfinite_supplemental_array(Tech):
    y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
    f = Tech.from_coeffs(jnp.stack([y, jnp.ones_like(y)], axis=-1))
    assert not f.isfinite()


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_predicates_jit_and_empty(Tech):
    def predicates(c):
        f = Tech.from_coeffs(c)
        return f.isfinite(), f.isinf(), f.isreal()

    compiled = jax.jit(predicates)
    for coefficients, expected in [
        ([0.0, 1.0], (True, False, True)),
        ([jnp.nan], (False, False, True)),
        ([jnp.inf], (False, True, True)),
        ([0j], (True, False, False)),
        ([1j], (True, False, False)),
    ]:
        assert tuple(bool(v) for v in compiled(jnp.asarray(coefficients))) == expected
    for f in (Tech.empty(), Tech.from_coeffs(jnp.asarray([]))):
        assert f.isfinite() and not f.isinf() and f.isreal()
