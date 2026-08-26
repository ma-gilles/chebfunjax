"""Port of MATLAB Chebfun tests/chebop/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _n(f, d):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopDiff:
    def test_all_matlab_assertions(self):
        d = (-3.0, -1.5)
        D = Chebop(lambda u: u.diff(), domain=d)
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x) ** 2 + 2), domain=d)
        assert _n(D * f - f.diff(), d) < TOL          # pass(1)
        L, res, is_linear = D.linearize()
        assert all(is_linear)                          # pass(2)

        D2 = jnp.pi * Chebop(lambda u: u.diff(2), domain=d)
        assert _n(D2 * f - jnp.pi * f.diff(2), d) < TOL  # pass(3)
        L, res, is_linear = D.linearize()
        assert all(is_linear)                          # pass(4)
