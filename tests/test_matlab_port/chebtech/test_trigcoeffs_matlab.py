"""Port of MATLAB Chebfun tests/chebtech/test_trigcoeffs.m (Opus 4.8).

MATLAB ``trigcoeffs(f)`` returns the trigonometric (Fourier) coefficients of a
chebtech.  chebfunjax now implements ``Chebtech{1,2}.trigcoeffs`` as a port of
``@chebtech/trigcoeffs.m`` (each Fourier mode is built as a tech of the same
kind and integrated against ``f``), so the scalar assertions run; the
array-valued assertions are skipped (chebfunjax Chebtech is scalar-valued).

Provenance
----------
MATLAB source : tests/chebtech/test_trigcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)

_SCALAR = "chebfunjax Chebtech is scalar-valued; no array-valued techs"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechTrigcoeffs:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_zeros(self, Tech, kind):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        p = f.trigcoeffs()
        assert _ninf(p) <= 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_constant(self, Tech, kind):
        # pass(n, 2)
        f = Tech.from_function(lambda x: 3 * jnp.ones_like(x))
        p = f.trigcoeffs()
        assert _ninf(p - 3) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len3(self, Tech, kind):
        # pass(n, 3): trigcoeffs(1+cos(pi x), 3) == [0.5 1 0.5]'.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = f.trigcoeffs(3)
        assert _ninf(p - jnp.array([0.5, 1.0, 0.5])) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len5(self, Tech, kind):
        # pass(n, 4): trigcoeffs(1+cos(pi x), 5) == [0 0.5 1 0.5 0]'.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = f.trigcoeffs(5)
        assert _ninf(p - jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_cos_len1(self, Tech, kind):
        # pass(n, 5): trigcoeffs(1+cos(pi x), 1) == 1.
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = f.trigcoeffs(1)
        assert _ninf(p - 1) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len5(self, Tech, kind):
        # pass(n, 6): 1 + exp(2i pi x) + exp(-i pi x), trigcoeffs 5 -> [0 1 1 0 1].
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        p = f.trigcoeffs(5)
        exact = jnp.array([0, 1, 1, 0, 1], dtype=jnp.complex128)
        assert _ninf(p - exact) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len9(self, Tech, kind):
        # pass(n, 7)
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        p = f.trigcoeffs(9)
        exact = jnp.array([0, 0, 0, 1, 1, 0, 1, 0, 0], dtype=jnp.complex128)
        assert _ninf(p - exact) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_complex_len3(self, Tech, kind):
        # pass(n, 8)
        f = Tech.from_function(
            lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x)
        )
        p = f.trigcoeffs(3)
        exact = jnp.array([1, 1, 0], dtype=jnp.complex128)
        assert _ninf(p - exact) < 10 * f.vscale * EPS

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
        f = Tech.from_function(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = f.trigcoeffs(0)
        assert p.shape[0] == 0
