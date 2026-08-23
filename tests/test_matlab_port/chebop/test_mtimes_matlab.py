"""Port of MATLAB Chebfun tests/chebop/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _n(f):
    xs = jnp.linspace(1e-9, math.pi - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopMtimes:
    def test_all_matlab_assertions(self):
        dom = (0.0, math.pi)
        u = cj.chebfun(jnp.sin, domain=dom)
        a = math.sqrt(2.0)
        N = Chebop(lambda u: u.diff(2) + u.cos(), domain=dom)
        assert _n((a * N) * u - a * (N * u)) < TOL   # err(1)
        assert _n((N * a) * u - a * (N * u)) < TOL   # err(2)

        N2 = Chebop(lambda x, u, v: [u.diff(2) + v.cos(),
                                     v.diff(2) - u.sin()], domain=dom)
        v = cj.chebfun(jnp.exp, domain=dom)
        r1 = (a * N2)(u, v)
        r2 = N2(u, v)
        for c1, c2 in zip(r1, r2):                   # err(3)/(4)
            assert _n(c1 - a * c2) < TOL
        try:                                         # err(6): N*N errors
            N * N
            assert False
        except TypeError:
            pass
