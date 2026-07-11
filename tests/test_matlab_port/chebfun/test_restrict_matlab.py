"""Port of MATLAB Chebfun tests/chebfun/test_restrict.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)


class TestChebfunRestrict:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_restrict_does_not_mutate(self):
        f = cj.chebfun(lambda x: x)
        _ = f.restrict(-0.5, 0.5)
        assert tuple(np.asarray(f.domain.breakpoints)) == (-1.0, 1.0)

    def test_restrict_values(self):
        f = cj.chebfun(jnp.sin)
        fr = f.restrict(-0.5, 0.5)
        xs = jnp.asarray(np.linspace(-0.49, 0.49, 100))
        err = jnp.abs(fr(xs) - jnp.sin(xs))
        assert float(jnp.max(err)) < 100 * EPS

    def test_restrict_piecewise(self):
        f = cj.chebfun(jnp.cos, domain=[-1.0, -0.3, 0.2, 1.0])
        fr = f.restrict(-0.5, 0.6)
        xs = jnp.asarray(np.linspace(-0.49, 0.59, 60))
        err = jnp.abs(fr(xs) - jnp.cos(xs))
        assert float(jnp.max(err)) < 100 * EPS
