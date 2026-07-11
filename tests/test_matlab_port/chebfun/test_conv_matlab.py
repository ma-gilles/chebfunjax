"""Port of MATLAB Chebfun tests/chebfun/test_conv.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_conv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunConv:
    def test_commutativity(self):
        f = cj.chebfun(lambda x: x)
        g = cj.chebfun(lambda x: jnp.sin(5 * x), domain=(2.0, 4.0))
        H1 = f.conv(g)
        H2 = g.conv(f)
        xs = jnp.asarray(np.linspace(1.2, 4.8, 60))
        err = jnp.abs(H1(xs) - H2(xs))
        assert float(jnp.max(err)) < 1e3 * EPS

    def test_gaussian_smoothing_mass(self):
        # convolution with a unit-mass kernel preserves total integral
        f = cj.chebfun(lambda x: jnp.exp(-x ** 2) /
                       float(np.sqrt(np.pi) * np.math.erf(1.0))
                       if False else jnp.exp(-x ** 2))
        g = cj.chebfun(lambda x: 1.0 + 0 * x, domain=(-0.5, 0.5))
        h = f.conv(g)
        assert abs(float(h.sum()) - float(f.sum()) * float(g.sum())) \
            < 1e-10
