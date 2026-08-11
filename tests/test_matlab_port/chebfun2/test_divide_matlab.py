"""Port of MATLAB Chebfun tests/chebfun2/test_divide.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): scalar division and the
left-division forms all reduce to Chebfun2 scalar arithmetic, which
exists.  MATLAB's ``2.\\f`` and ``2\\f`` (left division by a scalar) are
both ``f / 2`` in Python.

Provenance
----------
MATLAB source : tests/chebfun2/test_divide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Divide:
    def test_scalar_division(self):
        # pass(1-4): f./2, f/2, 2.\f and 2\f all equal cos(x*y)/2.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) / 2)
        assert float((f / 2 - g).norm()) < TOL
        assert float((f / 2.0 - g).norm()) < TOL

    def test_division_by_nonvanishing_chebfun2(self):
        # Dividing by a Chebfun2 that stays away from zero reproduces the
        # pointwise quotient.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        d = Chebfun2.from_function(lambda x, y: 3.0 + x * y)
        q = f / d
        rng = np.random.default_rng(0)
        x = 2 * rng.random(50) - 1
        y = 2 * rng.random(50) - 1
        exact = np.cos(x * y) / (3.0 + x * y)
        got = np.asarray(q(jnp.asarray(x), jnp.asarray(y)))
        assert float(np.max(np.abs(got - exact))) < TOL

    def test_scalar_over_chebfun2(self):
        # 2 / f is the reciprocal scaled by 2 where f does not vanish.
        d = Chebfun2.from_function(lambda x, y: 3.0 + x * y)
        r = 2.0 / d
        rng = np.random.default_rng(1)
        x = 2 * rng.random(50) - 1
        y = 2 * rng.random(50) - 1
        exact = 2.0 / (3.0 + x * y)
        got = np.asarray(r(jnp.asarray(x), jnp.asarray(y)))
        assert float(np.max(np.abs(got - exact))) < TOL
