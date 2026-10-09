"""Port of MATLAB Chebfun tests/diskfun/test_constructor.m (Fable 5).

Diskfun convention: f(theta, r) polar, theta in [-pi,pi], r in [0,1].

Provenance
----------
MATLAB source : tests/diskfun/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

TOL = 1e-10


class TestDiskfunConstructor:
    def test_smooth_construction_accuracy(self):
        def f(t, r):
            x = r * jnp.cos(t)
            y = r * jnp.sin(t)
            return jnp.exp(-2 * (x ** 2 + y ** 2)) + x * y
        g = Diskfun.from_function(f)
        rng = np.random.default_rng(7)
        t = jnp.asarray(rng.uniform(-np.pi, np.pi, 40))
        r = jnp.asarray(rng.uniform(0.05, 0.95, 40))
        err = jnp.abs(g(t, r) - f(t, r))
        assert float(jnp.max(err)) < 100 * TOL

    def test_mixed_order_modes_resolved(self):
        # the coarse-grid aliasing regression fixed in the Opus session
        def f(t, r):
            return r ** 2 * jnp.cos(2 * t) + r ** 6 * jnp.cos(6 * t)
        g = Diskfun.from_function(f)
        t0, r0 = jnp.asarray(0.7), jnp.asarray(0.8)
        assert abs(float(g(t0, r0)) - float(f(t0, r0))) < TOL
