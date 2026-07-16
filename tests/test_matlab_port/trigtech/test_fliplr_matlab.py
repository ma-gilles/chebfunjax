"""Port of MATLAB Chebfun tests/trigtech/test_fliplr.m (Opus 4.8).

fliplr reverses the column order of an array-valued trigtech; on a
scalar-valued trigtech it is the identity.  Array-valued trigtechs are now
supported (FIXED, Fable 5, Big-Three array-valued epic).

Provenance
----------
MATLAB source : tests/trigtech/test_fliplr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechFliplr:
    def test_scalar_identity(self):
        # pass(1): isequal(f, fliplr(f)) for scalar cos(pi x).
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Trigtech.from_function(lambda x: jnp.cos(jnp.pi * x))
        assert _ninf(f.fliplr().coeffs - f.coeffs) == 0.0

    def test_swap_columns(self):
        # pass(2): fliplr([sin cos]) == [cos sin].
        # FIXED (Fable 5, Big-Three array-valued epic): fliplr reverses columns.
        f = Trigtech.from_function(
            lambda x: jnp.stack([jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1)
        )
        g = Trigtech.from_function(
            lambda x: jnp.stack([jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)], axis=-1)
        )
        assert f.fliplr().coeffs.shape == g.coeffs.shape
        assert _ninf(f.fliplr().coeffs - g.coeffs) == 0.0
