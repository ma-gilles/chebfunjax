"""Port of MATLAB Chebfun tests/chebtech/test_length.m (Opus 4.8).

MATLAB ``length(f)`` returns ``size(f.coeffs, 1)`` (the number of
coefficient rows).  chebfunjax's faithful equivalent is ``len(f)``, which
equals ``f.coeffs.shape[0]``.  The ``fixedLength = 101`` case maps to
constructing at fixed length 101.  The array-valued case has no scalar
analogue and is skipped.

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
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_length_fixed_101(self, Tech):
        # pass(n,3): fixedLength=101 -> length(f) == 101
        f = Tech.from_function(jnp.sin, n=101)
        assert len(f) == 101
