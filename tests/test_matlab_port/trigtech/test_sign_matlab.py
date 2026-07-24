"""Port of MATLAB Chebfun tests/trigtech/test_sign.m (Opus 4.8[1m]).

sign(f) returns the signum of a root-free trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechSign:
    def test_positive_function(self):
        # sign(sin(pi x) + 2) == +1 everywhere.
        f = _tt(lambda x: jnp.sin(jnp.pi * x) + 2)
        h = f.sign()
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))
        assert _ninf(h(xx) - 1) < 10 * EPS

    def test_negative_function(self):
        f = _tt(lambda x: -(jnp.sin(jnp.pi * x) + 2))
        h = f.sign()
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))
        assert _ninf(h(xx) + 1) < 10 * EPS

    def test_complex_valued_function(self):
        # sign(exp(i pi x)) = exp(i pi x) (already on the unit circle).
        f = _tt(lambda x: jnp.exp(1j * jnp.pi * x))
        h = f.sign()
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))
        assert _ninf(h(xx) - f(xx)) < 10 * EPS

    def test_complex_array_valued(self):
        def fun(x):
            base = 2 + jnp.sin(jnp.pi * x)
            e = jnp.exp(1j * jnp.pi * x)
            return jnp.stack([base * e, -base * e, base + 0j], axis=-1)
        f = _tt(fun)
        xx = jnp.asarray(np.linspace(-0.95, 0.97, 100))
        ff = jnp.asarray(f(xx))
        gg = ff / jnp.abs(ff)
        h = f.sign()
        assert _ninf(jnp.asarray(h(xx)) - gg) < 1e3 * EPS
