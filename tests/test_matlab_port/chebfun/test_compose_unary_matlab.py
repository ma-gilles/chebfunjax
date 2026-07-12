"""Port of MATLAB Chebfun tests/chebfun/test_compose_unary.m
(Fable 5).

FIXED: Chebfun.compose(op) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_compose_unary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.97, 0.97, 100))


class TestChebfunComposeUnary:
    def test_smooth(self):
        f = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)))
        h = f.compose(jnp.exp)
        assert float(jnp.max(jnp.abs(h(XS) - jnp.exp(f(XS))))) \
            < 1e-13

    def test_non_default_domain(self):
        f = cj.chebfun(lambda x: jnp.sin(x - 0.1), domain=(-2, 7))
        xs = jnp.asarray(np.linspace(-1.95, 6.95, 100))
        h = f.compose(lambda v: jnp.tanh(2 * v))
        assert float(jnp.max(jnp.abs(
            h(xs) - jnp.tanh(2 * f(xs))))) < 1e-13
