"""Tests for randnfun (smooth band-limited random function, Opus 4.8)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class TestRandnfun:
    def test_band_limited_and_periodic(self):
        from chebfunjax.utils.randnfun import randnfun
        f = randnfun(0.2, key=jax.random.PRNGKey(3))
        # band-limited -> short representation
        assert len(f.funs[0].tech.coeffs) < 60
        # periodic
        assert abs(float(f(jnp.array(-1.0)) - f(jnp.array(1.0)))) < 1e-10
        # O(1) amplitude
        xs = np.linspace(-0.95, 0.95, 200)
        v = np.asarray(f(jnp.asarray(xs)))
        assert 0.1 < v.std() < 5.0

    def test_deterministic_given_key(self):
        from chebfunjax.utils.randnfun import randnfun
        f1 = randnfun(0.2, key=jax.random.PRNGKey(7))
        f2 = randnfun(0.2, key=jax.random.PRNGKey(7))
        assert abs(float(f1(jnp.array(0.3)) - f2(jnp.array(0.3)))) < 1e-14

    def test_gallerytrig_random(self):
        from chebfunjax.utils.gallerytrig import gallerytrig
        f = gallerytrig("random")
        assert np.isfinite(float(f(jnp.array(0.5))))
        g = gallerytrig("noisyfun")
        assert np.isfinite(float(g(jnp.array(0.5))))
