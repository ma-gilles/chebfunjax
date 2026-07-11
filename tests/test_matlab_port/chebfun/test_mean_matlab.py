"""Port of MATLAB Chebfun tests/chebfun/test_mean.m (Fable 5).

MATLAB's two-argument mean(f, g) has no counterpart ((f+g)/2 covers
the semantics); the scalar mean assertions are ported.

Provenance
----------
MATLAB source : tests/chebfun/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

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
        pytest.skip("chebfunjax has no array-valued chebfun")
