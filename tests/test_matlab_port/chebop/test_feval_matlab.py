"""Port of MATLAB Chebfun tests/chebop/test_feval.m (Fable 5).

MATLAB chebmatrix-wrapped arguments map to Python lists.

Provenance
----------
MATLAB source : tests/chebop/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-9


def _n(f):
    xs = jnp.linspace(1e-9, math.pi - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopFeval:
    def test_all_matlab_assertions(self):
        dom = (0.0, math.pi)
        x = cj.chebfun(lambda t: t, domain=dom)
        u = x.sin()
        Nu = u.diff(2) + u.cos()

        N = Chebop(lambda u: u.diff(2) + u.cos(), domain=dom)
        assert _n(N.feval(u) - Nu) < TOL
        assert _n(N(u) - Nu) < TOL

        N = Chebop(lambda x, u: u.diff(2) + u.cos(), domain=dom)
        assert _n(N.feval(u) - Nu) < TOL
        assert _n(N(u) - Nu) < TOL
        assert _n(N * u - Nu) < TOL
        assert _n(N.feval(x, u) - Nu) < TOL
        assert _n(N(x, u) - Nu) < TOL

        # chebmatrix-wrapped argument -> list
        N = Chebop(lambda u: u.diff(2) + u.cos(), domain=dom)
        assert _n(N([u]) - Nu) < TOL
        N = Chebop(lambda x, u: u.diff(2) + u.cos(), domain=dom)
        assert _n(N([u]) - Nu) < TOL
        assert _n(N * [u] - Nu) < TOL

        # system operator
        N = Chebop(lambda x, u, v: [u.diff(2) + v.cos(),
                                    v.diff(2) - u.sin()], domain=dom)
        v = x.exp()
        Nu1 = u.diff(2) + v.cos()
        Nu2 = v.diff(2) - u.sin()
        out = N(u, v)
        assert _n(out[0] - Nu1) < TOL and _n(out[1] - Nu2) < TOL
        out = N(x, u, v)
        assert _n(out[0] - Nu1) < TOL and _n(out[1] - Nu2) < TOL
        out = N([u, v])
        assert _n(out[0] - Nu1) < TOL and _n(out[1] - Nu2) < TOL
