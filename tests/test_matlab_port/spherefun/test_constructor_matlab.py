"""Port of MATLAB Chebfun tests/spherefun/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-11


class TestSpherefunConstructor:
    def test_smooth_construction_accuracy(self):
        def f(lam, th):
            x = jnp.cos(lam) * jnp.sin(th)
            y = jnp.sin(lam) * jnp.sin(th)
            z = jnp.cos(th)
            return jnp.exp(x) + jnp.sin(y * z)
        g = Spherefun.from_function(f)
        rng = np.random.default_rng(5)
        lam = jnp.asarray(rng.uniform(-np.pi, np.pi, 40))
        th = jnp.asarray(rng.uniform(0.05, np.pi - 0.05, 40))
        err = jnp.abs(g(lam, th) - f(lam, th))
        assert float(jnp.max(err)) < 1e3 * TOL

    def test_mixed_order_harmonics_resolved(self):
        # the coarse-grid aliasing regression fixed in the Opus session
        def f(lam, th):
            Y21 = Spherefun.sphharm(2, -1)
            Y43 = Spherefun.sphharm(4, -3)
            return Y21(lam, th) + Y43(lam, th)
        g = Spherefun.from_function(f)
        lam, th = jnp.asarray(0.3), jnp.asarray(1.4)
        assert abs(float(g(lam, th)) - float(f(lam, th))) < 1e-10
