"""Port of MATLAB Chebfun tests/chebfun/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_power.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

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
        pytest.skip("chebfunjax has no array-valued chebfun")
