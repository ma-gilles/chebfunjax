"""Port of MATLAB Chebfun tests/chebfun/test_inv.m (Fable 5).

FIXED: Chebfun.inv (compositional inverse) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_inv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunInv:
    def test_inverse_of_exp_is_log(self):
        f = cj.chebfun(jnp.exp, domain=(0.0, 1.0))
        g = f.inv()
        ys = jnp.asarray(np.linspace(1.05, np.e - 0.05, 20))
        np.testing.assert_allclose(np.asarray(g(ys)),
                                   np.log(np.asarray(ys)), atol=1e-10)

    def test_roundtrip(self):
        f = cj.chebfun(lambda x: x + 0.3 * jnp.sin(x),
                       domain=(-1.0, 1.0))
        g = f.inv()
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 15))
        np.testing.assert_allclose(
            np.asarray(g(f(xs))), np.asarray(xs), atol=1e-9)
