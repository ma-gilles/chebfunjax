"""Port of MATLAB Chebfun tests/chebfun2/test_gradys_function1.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_gradys_function1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e2 * EPS


class TestGradysFunction1:
    def test_bump_construction_accuracy(self):
        def g(x, y):
            d = 1 - ((x - 0.02) ** 2 + (y - 0.033) ** 2)
            return jnp.exp(-1.0 / jnp.maximum(d, 1e-300)) * (d > 0)
        f = Chebfun2.from_function(g, domain=(-np.pi, np.pi, -np.pi, np.pi))
        xs = jnp.asarray(np.linspace(-np.pi, np.pi, 101))
        xx, yy = jnp.meshgrid(xs, xs)
        err = g(xx, yy) - f(xx, yy)
        assert float(jnp.max(jnp.abs(err))) < 2e3 * TOL
