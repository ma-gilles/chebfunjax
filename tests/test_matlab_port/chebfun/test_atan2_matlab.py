"""Port of MATLAB Chebfun tests/chebfun/test_atan2.m (Fable 5).

FIXED: module-level atan2 exists; port flipped from a skip stub in
the Fable 5 audit.  A later fix resolved eps-level accuracy: atan2's
pieces are built by nudging endpoint samples off the roots of y (the
branch cut), emulating MATLAB's pref.techPrefs.extrapolate = true, so
the constructor no longer sees a spurious 2*pi jump at each break.

Provenance
----------
MATLAB source : tests/chebfun/test_atan2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

A = -2.25 * np.pi
B = 2.25 * np.pi
XX = jnp.asarray(np.linspace(0.99 * A, 0.99 * B, 100))


class TestChebfunAtan2:
    def test_scalar_valued_both_orders(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(
                lambda x: 0.5 + jnp.sin(x) * jnp.exp(-0.1 * x ** 2),
                domain=(A, B))
            g = cj.chebfun(lambda x: jnp.cos(x) * (1 + x ** 2),
                           domain=(A, B))
            h = cj.atan2(f, g)
        ff = np.asarray(f(XX))
        gg = np.asarray(g(XX))
        # pass(1)
        assert np.max(np.abs(np.asarray(h(XX))
                             - np.arctan2(ff, gg))) < 1e3 * 10 \
            * np.finfo(float).eps

        # pass(3): reversed arguments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h2 = cj.atan2(g, f)
        vs = max(np.max(np.abs(np.asarray(h2(XX)))), 1.0)
        assert np.max(np.abs(np.asarray(h2(XX))
                             - np.arctan2(gg, ff))) \
            < 1e4 * np.finfo(float).eps * vs
