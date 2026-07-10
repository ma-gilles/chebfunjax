"""Port of MATLAB Chebfun tests/chebtech/test_trigcoeffs.m (Opus 4.8).

MATLAB ``trigcoeffs(f)`` returns the trigonometric (Fourier) coefficients of a
chebtech.  chebfunjax's Chebtech has NO ``trigcoeffs`` method, so every scalar
assertion is marked xfail with a precise reason; the array-valued assertions
are skipped (chebfunjax Chebtech is scalar-valued).

Provenance
----------
MATLAB source : tests/chebtech/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_NO_TRIG = "chebfunjax Chebtech lacks trigcoeffs"
_SCALAR = "chebfunjax Chebtech is scalar-valued; no array-valued techs (and lacks trigcoeffs)"


class TestChebtechTrigcoeffs:
    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_zeros(self, Tech, kind):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        f.trigcoeffs()

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_constant(self, Tech, kind):
        # pass(n, 2)
        f = Tech.from_function(lambda x: 3 * jnp.ones_like(x))
        f.trigcoeffs()

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len3(self, Tech, kind):
        # pass(n, 3): trigcoeffs(1+cos(pi x), 3) == [0.5 1 0.5]'.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        f.trigcoeffs(3)

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len5(self, Tech, kind):
        # pass(n, 4): trigcoeffs(1+cos(pi x), 5) == [0 0.5 1 0.5 0]'.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        f.trigcoeffs(5)

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len1(self, Tech, kind):
        # pass(n, 5): trigcoeffs(1+cos(pi x), 1) == 1.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        f.trigcoeffs(1)

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len5(self, Tech, kind):
        # pass(n, 6): 1 + exp(2i pi x) + exp(-i pi x), trigcoeffs 5.
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        f.trigcoeffs(5)

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len9(self, Tech, kind):
        # pass(n, 7)
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        f.trigcoeffs(9)

    @pytest.mark.xfail(reason=_NO_TRIG, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len3(self, Tech, kind):
        # pass(n, 8)
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        f.trigcoeffs(3)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued_len5(self, Tech, kind):
        # pass(n, 9)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued_len7(self, Tech, kind):
        # pass(n, 10)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued_len3(self, Tech, kind):
        # pass(n, 11)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued_len0(self, Tech, kind):
        # pass(n, 12): trigcoeffs(f, 0) is empty.
        pytest.skip(_SCALAR)
