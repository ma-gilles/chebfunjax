"""Port of MATLAB Chebfun tests/spherefun/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-10


class TestSpherefunFeval:
    def test_pointwise_and_vectorized(self):
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(th) + jnp.sin(lam) * jnp.sin(th))
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0))) - 1.0) \
            < TOL
        lam = jnp.asarray(np.linspace(-3.0, 3.0, 25))
        th = jnp.asarray(np.linspace(0.1, 3.0, 25))
        exact = jnp.cos(th) + jnp.sin(lam) * jnp.sin(th)
        err = jnp.abs(f(lam, th) - exact)
        assert float(jnp.max(err)) < 100 * TOL
