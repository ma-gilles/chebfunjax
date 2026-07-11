"""Port of MATLAB Chebfun tests/diskfun/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-10


class TestDiskfunFeval:
    def test_center_and_boundary(self):
        f = Diskfun.from_function(lambda t, r: 1 - r ** 2)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0))) - 1.0) \
            < TOL
        assert abs(float(f(jnp.asarray(1.0), jnp.asarray(1.0)))) < TOL

    def test_vectorized(self):
        f = Diskfun.from_function(
            lambda t, r: r * jnp.cos(t) + r ** 2 * jnp.sin(2 * t))
        t = jnp.asarray(np.linspace(-3.0, 3.0, 30))
        r = jnp.asarray(np.linspace(0.05, 0.95, 30))
        exact = r * jnp.cos(t) + r ** 2 * jnp.sin(2 * t)
        err = jnp.abs(f(t, r) - exact)
        assert float(jnp.max(err)) < 100 * TOL
