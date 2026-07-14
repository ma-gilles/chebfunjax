"""Port of MATLAB Chebfun tests/chebtech/test_isinf.m (Fable 5).

chebfunjax has no ``isinf()`` method, but MATLAB ``@chebtech/isinf.m`` is
exactly ``out = any(isinf(f.coeffs(:)))`` -- a global scalar reduction over
the coefficients.  These tests construct genuine scalar and array-valued
(n, m) techs (MATLAB duplicates the scalar case for its "array-valued" check;
we port it as a real array-valued tech to exercise (n, m) construction) and
assert that equivalent.

Provenance
----------
MATLAB source : tests/chebtech/test_isinf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _isinf(f):
    """MATLAB @chebtech/isinf.m equivalent: any coeff infinite."""
    return bool(jnp.any(jnp.isinf(f.coeffs)))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsinf:
    def test_scalar_inf_is_inf(self, Tech):
        # pass(n,1): isinf(make({[], y})) with y(4) = inf
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        f = Tech.from_values(y)
        assert _isinf(f)

    def test_array_inf_is_inf(self, Tech):
        # pass(n,2): isinf of an array-valued tech with an inf column
        # FIXED (Fable 5, Big-Three array-valued epic).
        y = jnp.ones(11, dtype=jnp.float64).at[3].set(jnp.inf)
        Y = jnp.stack([y, jnp.ones(11, dtype=jnp.float64)], axis=-1)
        f = Tech.from_values(Y)
        assert _isinf(f)

    def test_finite_scalar_not_inf(self, Tech):
        # pass(n,3): ~isinf(make(@(x) x))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: x)
        assert not _isinf(f)

    def test_finite_array_not_inf(self, Tech):
        # pass(n,4): ~isinf(make(@(x) [x, x]))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([x, x], axis=-1))
        assert not _isinf(f)
