"""Port of MATLAB Chebfun tests/chebfun/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 100))


def _nrm(f):
    return float(jnp.max(jnp.abs(f(X))))


class TestChebfunPower:
    def test_power_zero(self):
        f = cj.chebfun(jnp.sin)
        g = f ** 0
        assert _nrm(g - 1) < 10 * EPS

    def test_power_one(self):
        f = cj.chebfun(jnp.sin)
        assert _nrm((f ** 1) - f) < 10 * EPS

    def test_power_two(self):
        f = cj.chebfun(jnp.sin)
        h = cj.chebfun(lambda x: jnp.sin(x) ** 2)
        assert _nrm((f ** 2) - h) < 10 * EPS

    def test_power_three(self):
        f = cj.chebfun(jnp.sin)
        h = cj.chebfun(lambda x: jnp.sin(x) ** 3)
        assert _nrm((f ** 3) - h) < 10 * EPS

    def test_array_valued(self):
        # pass(5, 7, 9, 11): array-valued f.^0, f.^1, f.^2, f.^3 (3 cols preserved).
        # FIXED (Fable 5, Big-Three array-valued epic): f**0 now preserves columns.
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), 1j * jnp.exp(x)], axis=-1))
        g0 = f ** 0
        assert g0(X).shape[-1] == 3 and _nrm(g0 - 1.0) < EPS
        g1 = f ** 1
        assert g1(X).shape[-1] == 3 and _nrm(g1 - f) < EPS
        h2 = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x) ** 2, jnp.cos(x) ** 2, -jnp.exp(2 * x)], axis=-1)
        )
        g2 = f ** 2
        assert g2(X).shape[-1] == 3 and _nrm(g2 - h2) < 10 * h2.vscale * EPS
        h3 = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x) ** 3, jnp.cos(x) ** 3, -1j * jnp.exp(3 * x)], axis=-1)
        )
        g3 = f ** 3
        assert g3(X).shape[-1] == 3 and _nrm(g3 - h3) < 10 * h3.vscale * EPS
