"""Port of MATLAB Chebfun tests/chebtech/test_alias.m (Opus 4.8).

MATLAB ``testclass.alias(coeffs, k)`` aliases a Chebyshev coefficient vector
down/up to length ``k``.  chebfunjax has NO ``alias`` static method, so the two
scalar assertions are xfail'd (lacks alias); the array-valued assertions are
skipped (chebfunjax Chebtech is scalar-valued).

Provenance
----------
MATLAB source : tests/chebtech/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_NO_ALIAS = "chebfunjax lacks chebtech.alias"
_SCALAR = "chebfunjax Chebtech is scalar-valued; no array-valued techs (and lacks alias)"


class TestChebtechAlias:
    @pytest.mark.xfail(reason=_NO_ALIAS, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_alias_up_to_11(self, Tech, kind):
        # pass(n, 1): alias(sin coeffs, 11) reproduces sin on the k=11 grid.
        f = Tech.from_function(jnp.sin)
        Tech.alias(f.coeffs, 11)

    @pytest.mark.xfail(reason=_NO_ALIAS, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_alias_down_to_2(self, Tech, kind):
        # pass(n, 2): alias(sin coeffs, 2) -> length 2 with aliased values.
        f = Tech.from_function(jnp.sin)
        Tech.alias(f.coeffs, 2)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin_up_to_11(self, Tech, kind):
        # pass(n, 3)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin_down_to_2(self, Tech, kind):
        # pass(n, 4)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_32(self, Tech, kind):
        # pass(n, 5)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_100(self, Tech, kind):
        # pass(n, 6)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_2(self, Tech, kind):
        # pass(n, 7)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_1(self, Tech, kind):
        # pass(n, 8)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_2(self, Tech, kind):
        # pass(n, 9)
        pytest.skip(_SCALAR)
