"""Port of MATLAB Chebfun tests/chebtech/test_fliplr.m (Fable 5).

MATLAB ``fliplr`` flips the *columns* of an array-valued tech: for a
scalar-valued tech ``fliplr(f) == f`` (identity), and for array-valued
techs it reverses the column order.  chebfunjax has both ``fliplr()``
and ``isequal()`` (same length and identical coefficients), so both
MATLAB assertions port directly with no gaps.

Provenance
----------
MATLAB source : tests/chebtech/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechFliplr:
    def test_fliplr_scalar_identity(self, Tech):
        # pass(n,1): isequal(f, fliplr(f)) for scalar sin
        f = Tech.from_function(jnp.sin)
        assert f.isequal(f.fliplr())

    def test_fliplr_array_reverses_columns(self, Tech):
        # pass(n,2): isequal(fliplr([sin cos]), [cos sin])
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.cos(x), jnp.sin(x)], axis=-1))
        assert f.fliplr().isequal(g)
