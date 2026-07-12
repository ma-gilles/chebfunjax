"""Port of MATLAB Chebfun tests/chebfun/test_compose_chebfuns.m
(Fable 5).

FIXED: f(g) chebfun-of-chebfun composition added in the Fable 5
audit (Chebfun.__call__ dispatches to compose_chebfun).

Provenance
----------
MATLAB source : tests/chebfun/test_compose_chebfuns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.97, 0.97, 100))


class TestChebfunComposeChebfuns:
    def test_two_smooth(self):
        # pass(1)
        f = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)))
        g = cj.chebfun(lambda x: jnp.sin(x - 0.1))
        h = f(g)
        exact = np.cos(2 * (np.sin(np.asarray(XS) - 0.1) + 0.2))
        assert np.max(np.abs(np.asarray(h(XS)) - exact)) < 1e-13

    def test_non_default_domain(self):
        # pass(2)
        f = cj.chebfun(lambda x: jnp.cos(2 * (x + 0.2)),
                       domain=(-2, 7))
        g = cj.chebfun(lambda x: 0.5 * (x + 1) + 1.0, domain=(-2, 7))
        h = f(g)
        xs = jnp.asarray(np.linspace(-1.95, 6.95, 100))
        exact = np.cos(2 * (0.5 * (np.asarray(xs) + 1) + 1.2))
        assert np.max(np.abs(np.asarray(h(xs)) - exact)) < 1e-12

    def test_nonsmooth_outer(self):
        # pass(3)-(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # root-based abs (what MATLAB abs.m does) rather than
            # bisection splitting
            f = cj.chebfun(
                lambda x: jnp.cos(2 * (x + 0.2))).abs()
            g = cj.chebfun(lambda x: jnp.sin(x - 0.1))
            h = f(g)
        exact = np.abs(np.cos(
            2 * (np.sin(np.asarray(XS) - 0.1) + 0.2)))
        assert np.max(np.abs(np.asarray(h(XS)) - exact)) < 1e-12
