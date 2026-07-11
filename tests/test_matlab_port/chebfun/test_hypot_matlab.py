"""Port of MATLAB Chebfun tests/chebfun/test_hypot.m (Fable 5).

FIXED: Chebfun.hypot added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_hypot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunHypot:
    def test_pythagorean_identity(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = f.hypot(g)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 30))
        np.testing.assert_allclose(np.asarray(h(xs)), 1.0, atol=1e-12)

    def test_scaling_robustness(self):
        # MATLAB's hypot example: no overflow for large components
        f = cj.chebfun(lambda x: 3e300 * (2 + jnp.sin(x)))
        g = cj.chebfun(lambda x: 4e300 * (2 + jnp.sin(x)))
        h = f.hypot(g)
        v = float(h(jnp.asarray(0.0)))
        assert np.isfinite(v)
        assert abs(v - 5e300 * 2) / 1e301 < 1e-10
