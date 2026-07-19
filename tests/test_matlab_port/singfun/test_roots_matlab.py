"""Port of MATLAB Chebfun tests/singfun/test_roots.m (Opus 4.8).

chebfunjax Singfun implements no ``roots`` method, so every assertion is
xfailed (the call raises ``AttributeError``).  The analytic exact roots from
the MATLAB test are preserved in each case for when rootfinding lands.

Provenance
----------
MATLAB source : tests/singfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

A = 0.56
B = -0.56
C = 1.28
D = -1.28

_REASON = "chebfunjax Singfun has no roots() method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunRoots:
    def test_frac_root_left(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        r = f.roots()
        assert _ninf(r - (-1.0)) < 10 * f.smoothPart.vscale * EPS

    def test_frac_pole_left_many_roots(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(50 * np.pi * x), (D + 1, 0.0))
        r = f.roots()
        r_exact = np.arange(-1 + 1 / 50, 1 + 1e-12, 1 / 50)
        assert _ninf(np.asarray(r) - r_exact) < 10 * f.smoothPart.vscale * EPS

    def test_frac_root_right(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        r = f.roots()
        assert _ninf(r - 1.0) < 10 * f.smoothPart.vscale * EPS

    def test_root_at_right_endpoint(self):
        f = _sf(lambda x: (1 - x) ** B * (jnp.exp(x) - np.exp(1)), (0.0, 1 + B))
        r = f.roots()
        assert _ninf(r - 1.0) < 10 * f.smoothPart.vscale * EPS

    def test_pole_and_root(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** C, (B, C))
        r = f.roots()
        assert _ninf(np.sort(np.asarray(r)) - np.array([0.0, 1.0])) < 10 * f.smoothPart.vscale * EPS

    def test_root_close_to_endpoint(self):
        p = 1 - 1e-14
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x - p) * (1 - x) ** B, (B, B))
        r = f.roots()
        assert _ninf(r - p) < 10 * f.smoothPart.vscale * EPS
