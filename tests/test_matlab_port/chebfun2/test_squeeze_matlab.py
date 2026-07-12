"""Port of MATLAB Chebfun tests/chebfun2/test_squeeze.m (Fable 5).

FIXED: Chebfun2.squeeze added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_squeeze.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 10 * np.finfo(float).eps


class TestChebfun2Squeeze:
    def test_no_squeeze(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = f.squeeze()
        assert isinstance(g, Chebfun2)
        xs = jnp.asarray(np.linspace(-1, 1, 9))
        xx, yy = jnp.meshgrid(xs, xs, indexing="ij")
        assert float(jnp.max(jnp.abs(f(xx, yy) - g(xx, yy)))) \
            < 2 * TOL

    def test_squeeze_constant_in_y(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x)).squeeze()
        xs = jnp.asarray(np.linspace(-1, 1, 21))
        assert float(jnp.max(jnp.abs(f(xs) - jnp.cos(xs)))) \
            < 100 * TOL

    def test_squeeze_constant_in_x(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(y), domain=(-1, 1, -2, 3)).squeeze()
        ys = jnp.asarray(np.linspace(-2, 3, 21))
        assert float(jnp.max(jnp.abs(f(ys) - jnp.cos(ys)))) < TOL * 10
