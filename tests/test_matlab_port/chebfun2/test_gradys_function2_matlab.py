"""Port of MATLAB Chebfun tests/chebfun2/test_gradys_function2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_gradys_function2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e2 * EPS


class TestGradysFunction2:
    def test_bump_construction_accuracy(self):
        def g(x, y):
            r2 = (x - 0.2) ** 2 + (y - 0.33) ** 2
            d = 1 - r2
            return jnp.exp(-r2 / jnp.maximum(d, 1e-300)) * (d > 0)
        f = Chebfun2.from_function(g, domain=(-np.pi, np.pi, -np.pi, np.pi))
        xs = jnp.asarray(np.linspace(-np.pi, np.pi, 101))
        xx, yy = jnp.meshgrid(xs, xs)
        err = g(xx, yy) - f(xx, yy)
        assert float(jnp.max(jnp.abs(err))) < 2e3 * TOL
