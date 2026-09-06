"""Port of MATLAB Chebfun tests/chebop/test_linearize.m (Fable 5).

``linearize(N)`` maps to ``N.linearize()`` (returns ``(L, res,
isLinear)``); ``linearize(N, u)`` maps to ``N.linearize(u)`` which
returns a :class:`LinearizedChebop` applied with ``L * v``.  MATLAB
``L.blocks{j}`` maps to ``L.blocks[i][j]`` (parameter unknowns collapse
to Chebfun coefficient blocks exactly as in MATLAB).

Provenance
----------
MATLAB source : tests/chebop/test_linearize.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def _n(f, d=(0.0, math.pi)):
    xs = jnp.linspace(d[0] + 1e-6, d[1] - 1e-6, 41)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopLinearize:
    def test_all_matlab_assertions(self):
        dom = (0.0, 1.0, math.pi)
        x = cj.chebfun(lambda t: t, domain=dom)
        u = x.sin()
        v = x.cos()
        w = (-x).exp()

        # err(1): linear op, L*u == diff(u)
        N = Chebop(lambda u: u.diff(), domain=dom)
        L, _res, _lin = N.linearize()
        assert _n((L * u)[0] - u.diff()) < TOL

        # err(2): variable coefficients
        N = Chebop(lambda x, u: u.diff(2) + u.diff() + x.sin() * u,
                   domain=dom)
        L, _res, _lin = N.linearize()
        assert _n((L * u)[0] - (u.diff(2) + u.diff() + x.sin() * u)) < TOL

        # err(3): u^2 about u -> 2 u v
        N = Chebop(lambda u: u ** 2, domain=dom)
        L = N.linearize(u)
        assert _n(L * v - 2.0 * u * v) < TOL

        # err(4): u'' + u^2 about u
        N = Chebop(lambda u: u.diff(2) + u ** 2, domain=dom)
        L = N.linearize(u)
        assert _n(L * v - (v.diff(2) + 2.0 * u * v)) < TOL

        # err(5): system about [u; v] applied to [w; v]
        N = Chebop(lambda x, u, v: [u.diff(2) + v ** 2,
                                    v.diff() - u.cos()], domain=dom)
        L = N.linearize([u, v])
        out = L * [w, v]
        assert _n(out[0] - (w.diff(2) + 2.0 * v ** 2)) < TOL
        assert _n(out[1] - (v.diff() + u.sin() * w)) < TOL

        # err(6)/(7): parameter unknown a -> Chebfun block
        N = Chebop(lambda x, u, a: u.diff(2) + a ** 2, domain=dom)
        L = N.linearize([cj.chebfun(lambda t: 0.0 * t, domain=dom), 1.0])
        assert isinstance(L.blocks[0][1], Chebfun)
        L0, _r, _l = N.linearize()
        assert isinstance(L0[0, 1], Chebfun)

        # err(8)/(9): two functions, two parameters
        N = Chebop(lambda x, u, v, a, b: [x * v + 0.001 * u.diff(2) + a + 2 * b,
                                          (a * v).diff() - u],
                   domain=(-1.0, 1.0))
        u0 = [cj.chebfun(jnp.sin), cj.chebfun(jnp.cos), 1.0, 0.0]
        L = N.linearize(u0)
        isf = [[isinstance(L.blocks[i][j], Chebfun) for j in range(4)]
               for i in range(2)]
        assert isf == [[False, False, True, True],
                       [False, False, True, True]]
        L0, _r, _l = N.linearize()
        isf0 = [[isinstance(L0[i, j], Chebfun) for j in range(4)]
                for i in range(2)]
        assert isf0 == [[False, False, False, True],
                        [False, False, False, True]]
