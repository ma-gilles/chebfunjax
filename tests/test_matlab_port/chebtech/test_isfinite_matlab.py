"""Port of MATLAB Chebfun tests/chebtech/test_isfinite.m (Fable 5).

chebfunjax has no ``isfinite()`` method, but MATLAB ``@chebtech/isfinite.m``
is exactly ``out = all(isfinite(f.coeffs(:)))`` -- a global scalar reduction
over the coefficients.  These tests construct genuine scalar and array-valued
(n, m) techs (MATLAB duplicates the scalar case for its "array-valued" check;
we port it as a real array-valued tech to exercise (n, m) construction) and
assert that equivalent.

Provenance
----------
MATLAB source : tests/chebtech/test_isfinite.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _isfinite(f):
    """MATLAB @chebtech/isfinite.m equivalent: all coeffs finite."""
    return bool(jnp.all(jnp.isfinite(f.coeffs)))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsfinite:
    def test_scalar_inf_not_finite(self, Tech):
        # pass(n,1): ~isfinite(make({[], y})) with y(4) = inf
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_values(y)
        assert not _isfinite(f)

    def test_array_inf_not_finite(self, Tech):
        # pass(n,2): ~isfinite of an array-valued tech with an inf column
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        Y = jnp.stack([y, jnp.ones(11, dtype=jnp.float64)], axis=-1)
        f = Tech.from_values(Y)
        assert not _isfinite(f)

    def test_finite_scalar_is_finite(self, Tech):
        # pass(n,3): isfinite(make(@(x) x))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: x)
        assert _isfinite(f)

    def test_finite_array_is_finite(self, Tech):
        # pass(n,4): isfinite(make(@(x) [x, x]))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([x, x], axis=-1))
        assert _isfinite(f)
