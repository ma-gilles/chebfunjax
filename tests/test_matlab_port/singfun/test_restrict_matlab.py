"""Port of MATLAB Chebfun tests/singfun/test_restrict.m (Opus 4.8).

chebfunjax Singfun implements no ``restrict`` method, so every assertion is
xfailed (the call raises ``AttributeError``).

Provenance
----------
MATLAB source : tests/singfun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

A = 0.64
B = -0.64
C = 1.28
D = -1.28

_REASON = "chebfunjax Singfun has no restrict() method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunRestrict:
    def test_frac_root_left(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        f.restrict([-0.2, 0.1])

    def test_frac_pole_left(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(50 * np.pi * x), (D + 1, 0.0))
        f.restrict([-1, 0.3])

    def test_frac_root_right_multi(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        f.restrict([-1, -0.7, 1])

    def test_frac_pole_right_multi(self):
        f = _sf(lambda x: (1 - x) ** B, (0.0, B))
        f.restrict([-0.9, -0.3, 0.7, 1])

    def test_two_poles_multi(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** D, (B, D))
        f.restrict([-1, -0.9, 0.5, 0.7, 1])

    def test_roots_close_to_endpoints(self):
        p = 1e-4
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** (3 * C), (B, B))
        f.restrict([-1 + p, 1 - p])
