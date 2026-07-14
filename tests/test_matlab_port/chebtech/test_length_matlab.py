"""Port of MATLAB Chebfun tests/chebtech/test_length.m (Opus 4.8).

MATLAB ``length(f)`` returns ``size(f.coeffs, 1)`` (the number of
coefficient rows).  chebfunjax's faithful equivalent is ``len(f)``, which
equals ``f.coeffs.shape[0]``.  The ``fixedLength = 101`` case maps to
constructing at fixed length 101.  The array-valued case (pass 2) is now
supported: (n, m) coeffs, one function per column, and ``len(f)`` is the row
count regardless of m (FIXED, Fable 5, Big-Three array-valued epic).

Provenance
----------
MATLAB source : tests/chebtech/test_length.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechLength:
    def test_length_matches_coeffs(self, Tech):
        # pass(n,1): length(f) == size(f.coeffs, 1) for scalar sin
        f = Tech.from_function(jnp.sin)
        assert len(f) == f.coeffs.shape[0]

    def test_length_array_valued(self, Tech):
        # pass(n,2): length(f) == size(f.coeffs, 1) for [sin cos 1i*exp]
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs; len(f)
        # is the row count (number of coefficients), independent of m.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1)
        )
        assert len(f) == f.coeffs.shape[0]

    def test_length_fixed_101(self, Tech):
        # pass(n,3): fixedLength=101 on [sin cos 1i*exp] -> length(f) == 101
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued fixed-length
        # construction (matches MATLAB's array-valued pass 3).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1),
            n=101,
        )
        assert len(f) == 101
