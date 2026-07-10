"""Port of MATLAB Chebfun tests/chebtech/test_size.m (Opus 4.8).

chebfunjax Chebtech has no ``size()`` method returning ``[n, ncols]``
(techs are scalar-valued, i.e. a single column).  The faithful scalar
equivalent of ``size(f) == size(f.coeffs)`` is that the tech's length
matches the number of coefficients (``f.n == f.coeffs.shape[0]``), and
the ``fixedLength = 101`` case maps to constructing at fixed length 101.
The array-valued (3-column) cases have no scalar analogue and are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechSize:
    def test_size_matches_coeffs(self, Tech):
        # pass(n,1): all(size(f) == size(f.coeffs)) for scalar sin
        f = Tech.from_function(jnp.sin)
        assert f.n == f.coeffs.shape[0]
        assert len(f) == f.coeffs.shape[0]

    def test_size_array_valued(self, Tech):
        # pass(n,2): all(size(f) == size(f.coeffs)) for [sin cos 1i*exp]
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
        )

    def test_size_fixed_length_101(self, Tech):
        # pass(n,3): fixedLength=101 -> size(f) == [101, 3].
        # Scalar analogue: fixed-length-101 tech has length 101 (the
        # 3-column count has no scalar-valued analogue).
        f = Tech.from_function(jnp.sin, n=101)
        assert f.n == 101
