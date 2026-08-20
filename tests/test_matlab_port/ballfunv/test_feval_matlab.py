"""Port of MATLAB Chebfun tests/ballfunv/test_feval.m (Fable 5).

MATLAB stacks the three component values in a trailing dimension;
chebfunjax's Ballfunv.__call__ returns the same values as a tuple.

Provenance
----------
MATLAB source : tests/ballfunv/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 2.220446049250313e-16


class TestBallfunvFeval:
    def test_all_matlab_assertions(self):
        zero = Ballfun.from_function(
            lambda r, lam, th: 0.0 * r, spherical=True)
        r = jnp.asarray(0.5)
        lam = jnp.asarray(float(np.pi))
        th = jnp.asarray(float(np.pi / 2))

        # Example 1: F = (1, 0, 0).
        f = Ballfun.from_function(lambda rr, ll, tt: 1.0 + 0.0 * rr, spherical=True)
        F = Ballfunv(f, zero, zero)
        vals = F(r, lam, th)
        want = [1.0, 0.0, 0.0]
        for v, w in zip(vals, want):
            assert abs(float(v) - w) < TOL

        # Example 2: F = (0, 0, r sin(th) cos(lam)).
        f = Ballfun.from_function(
            lambda rr, ll, tt: rr * jnp.sin(tt) * jnp.cos(ll),
            spherical=True)
        F = Ballfunv(zero, zero, f)
        vals = F(r, lam, th)
        want = [0.0, 0.0, -0.5]
        for v, w in zip(vals, want):
            assert abs(float(v) - w) < TOL
