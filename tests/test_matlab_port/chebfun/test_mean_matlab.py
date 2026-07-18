"""Port of MATLAB Chebfun tests/chebfun/test_mean.m (Fable 5).

MATLAB's two-argument mean(f, g) has no counterpart ((f+g)/2 covers
the semantics); the scalar and array-valued single-argument mean
assertions are ported (array-valued mean returns a per-column (m,) vector).

Provenance
----------
MATLAB source : tests/chebfun/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 60))


class TestChebfunMean:
    def test_mean_of_two_functions(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = cj.chebfun(lambda x: 0.5 * (jnp.sin(x) + jnp.cos(x)))
        d = 0.5 * (f + g) - h
        assert float(jnp.max(jnp.abs(d(X)))) < 10 * EPS

    def test_mean_of_odd_function_is_zero(self):
        f = cj.chebfun(jnp.sin)
        assert abs(float(f.mean())) < EPS * 10

    def test_array_valued(self):
        # pass(5): mean of [sin(x), x] is ~0 per column.
        # pass(7): mean == sum / length on the domain [0, 6].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued mean -> (m,).
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), x], axis=-1))
        assert float(jnp.max(jnp.abs(f.mean()))) < EPS
        f2 = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), x], axis=-1), domain=(0, 6))
        assert float(jnp.max(jnp.abs(f2.mean() - f2.sum() / 6))) < f2.vscale * EPS
