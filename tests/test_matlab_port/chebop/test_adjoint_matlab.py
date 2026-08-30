"""Port of MATLAB Chebfun tests/chebop/test_adjoint.m (Fable 5).

MATLAB's ``adjoint(L)`` / ``L'`` map to ``L.adjoint()``.

Provenance
----------
MATLAB source : tests/chebop/test_adjoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


def _n(f, d=(-1.0, 1.0)):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def _ip(u, v):
    return float(jnp.asarray(u.inner(v)))


class TestChebopAdjoint:
    def test_second_order_selfadjoint(self):
        x = cj.chebfun(lambda t: t)
        c = (jnp.pi * x ** 2).sin()
        u = (x + 1.0) * (x - 1.0) * x.exp()
        nrm = float(jnp.sqrt(jnp.asarray(u.inner(u))))
        L = Chebop(lambda x, u: (-1.0) * u.diff(2) + c * u)
        L.lbc = 0.0
        L.rbc = 0.0
        Ls = L.adjoint()
        assert _n((-1.0) * u.diff(2) + c * u - Ls(u)) < nrm * TOL  # 1
        assert Ls.lbc == 0.0 and Ls.rbc == 0.0                     # 2/3
        assert abs(_ip(u, L * u) - _ip(Ls * u, u)) < nrm * TOL     # 4

    def test_first_order(self):
        x = cj.chebfun(lambda t: t)
        u = x.exp()
        v = (x + 1.0) * (x - 1.0) * x.exp()
        nrm = 10.0
        L = Chebop(lambda u: u.diff())
        Ls = L.adjoint()
        assert _n((-1.0) * v.diff() - Ls * v) < nrm * TOL          # 8
        assert Ls.lbc == 0.0 and Ls.rbc == 0.0                     # 9/10
        assert abs(_ip(v, L * u) - _ip(Ls * v, u)) < nrm * TOL     # 11

    def test_first_order_system(self):
        x = cj.chebfun(lambda t: t)
        u1 = (x + 1.0) * x.exp()
        u2 = (x + 1.0) * x.sin()
        v1 = (x - 1.0) * x.exp()
        v2 = (x - 1.0) * x.sin()
        nrm = 20.0
        L = Chebop(lambda x, u1, u2: [u1.diff() + u2, u1 + u2.diff()])
        L.lbc = lambda u1, u2: [u1, u2]
        Ls = L.adjoint()
        out = Ls([v1, v2])
        want = [(-1.0) * v1.diff() + v2, v1 - v2.diff()]
        assert _n(out[0] - want[0]) < nrm * TOL                    # 12
        assert _n(out[1] - want[1]) < nrm * TOL
        assert Ls._rbc_raw is not None and Ls._lbc_raw is None     # 13
        Lu = L([u1, u2])
        lhs = _ip(v1, Lu[0]) + _ip(v2, Lu[1])
        rhs = _ip(out[0], u1) + _ip(out[1], u2)
        assert abs(lhs - rhs) < nrm * TOL                          # 14
