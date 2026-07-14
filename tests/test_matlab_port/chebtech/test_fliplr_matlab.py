"""Port of MATLAB Chebfun tests/chebtech/test_fliplr.m (Opus 4.8).

MATLAB ``fliplr`` flips the *columns* of an array-valued tech.  For a
scalar-valued tech ``fliplr(f) == f`` (identity), and for array-valued
techs it reverses column order.  chebfunjax has NO ``fliplr()`` method,
is scalar-valued, and also has no ``isequal()`` predicate to express the
MATLAB checks, so both assertions are skipped with a precise reason.  No
assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

# FIXED (Fable 5, Big-Three array-valued epic): fliplr added with the
# (n, m) coefficient support; isequal maps to exact coeff equality.


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechFliplr:
    def test_fliplr_scalar_identity(self, Tech):
        # pass(n,1): isequal(f, fliplr(f)) for scalar sin
        f = Tech.from_function(jnp.sin)
        assert np.array_equal(np.asarray(f.coeffs),
                              np.asarray(f.fliplr().coeffs))

    def test_fliplr_array_reverses_columns(self, Tech):
        # pass(n,2): isequal(fliplr([sin cos]), [cos sin])
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.cos(x), jnp.sin(x)], axis=-1))
        assert np.allclose(np.asarray(f.fliplr().coeffs),
                           np.asarray(g.coeffs), atol=0, rtol=0)
