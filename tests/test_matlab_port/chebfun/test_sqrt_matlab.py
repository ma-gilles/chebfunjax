"""Port of MATLAB Chebfun tests/chebfun/test_sqrt.m (Fable 5).

Positive-function square roots on [-2, 7] at MATLAB tolerances;
root-touching/singular cases skipped (need blowup at the chebfun level).

Provenance
----------
MATLAB source : tests/chebfun/test_sqrt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(np.sort(9 * RNG.uniform(size=100) - 2))


class TestChebfunSqrt:
    def test_runge_reciprocal(self):
        f = cj.chebfun(lambda x: 1.0 / (1 + 25 * x ** 2),
                       domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = 1.0 / jnp.sqrt(1 + 25 * X ** 2)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e2 * EPS * float(jnp.max(exact))

    def test_oscillatory_positive(self):
        f = cj.chebfun(lambda x: jnp.sin(50 * x) ** 2 + 1,
                       domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = jnp.sqrt(jnp.sin(50 * X) ** 2 + 1)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * float(jnp.max(exact))

    def test_root_touching(self):
        pytest.skip("sqrt of a function with roots needs chebfun-level "
                    "singular exponents (blowup)")
