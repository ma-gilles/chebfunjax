"""Port of MATLAB Chebfun tests/chebfun/test_flipud.m (Fable 5).

flipud(f)(x) = f(-x) (reversal about the domain midpoint).

Provenance
----------
MATLAB source : tests/chebfun/test_flipud.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunFlipud:
    def test_flip_reverses_argument(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.flipud()
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        err = jnp.abs(g(xs) - f(-xs))
        assert float(jnp.max(err)) < 100 * EPS * f.vscale

    def test_double_flip_is_identity(self):
        f = cj.chebfun(lambda x: jnp.cos(5 * x) + x)
        g = f.flipud().flipud()
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        err = jnp.abs(g(xs) - f(xs))
        assert float(jnp.max(err)) < 100 * EPS * f.vscale
