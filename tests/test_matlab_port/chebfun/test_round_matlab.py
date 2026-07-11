"""Port of MATLAB Chebfun tests/chebfun/test_round.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_round.m
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


class TestChebfunRound:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_round_of_sin(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        g = f.round()
        # MATLAB round vs numpy: both round-half-away here is
        # immaterial since sin never hits +-0.5 at the test points.
        exact = jnp.round(jnp.sin(X))
        mask = jnp.abs(jnp.abs(jnp.sin(X)) - 0.5) > 1e-3
        err = jnp.abs(g(X) - exact)[mask]
        assert float(jnp.max(err)) <= 10 * max(g.vscale, 1.0) * EPS

    def test_round_of_exp(self):
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.5, 1.0])
        g = f.round()
        exact = jnp.round(jnp.exp(X))
        mask = jnp.abs(jnp.exp(X) - jnp.round(jnp.exp(X))) < 0.499
        err = jnp.abs(g(X) - exact)[mask]
        assert float(jnp.max(err)) <= 10 * max(g.vscale, 1.0) * EPS
