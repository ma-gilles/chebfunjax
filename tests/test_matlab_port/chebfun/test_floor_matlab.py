"""Port of MATLAB Chebfun tests/chebfun/test_floor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_floor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunFloor:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_floor_of_sin(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        g = f.floor()
        exact = jnp.sign(X) / 2 - 0.5   # floor(sin x) on [-1,1]
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) <= 10 * max(g.vscale, 1.0) * EPS

    def test_floor_of_exp(self):
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.5, 1.0])
        g = f.floor()
        exact = jnp.floor(jnp.exp(X))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) <= 10 * max(g.vscale, 1.0) * EPS
