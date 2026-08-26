"""Port of MATLAB Chebfun tests/chebop/test_stringConstructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_stringConstructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-9


def _n(f, d=(0.0, 5.0)):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def _solve(L):
    L.lbc = 1.0
    L.rbc = 2.0
    return L.solve(0.0)


class TestChebopStringConstructor:
    def test_all_matlab_assertions(self):
        # pass(1): u'' + u
        u1 = _solve(Chebop("u``+u", domain=(0.0, 5.0)))
        u2 = _solve(Chebop(lambda u: u.diff(2) + u, domain=(0.0, 5.0)))
        assert _n(u1 - u2) < TOL

        # pass(2): u'' + x*u
        u1 = _solve(Chebop("u``+x*u", domain=(0.0, 5.0)))
        u2 = _solve(Chebop(lambda x, u: u.diff(2) + x * u,
                           domain=(0.0, 5.0)))
        assert _n(u1 - u2) < TOL

        # pass(3): nonlinear u'' + sin(u)
        u1 = _solve(Chebop("u``+sin(u)", domain=(0.0, 5.0)))
        u2 = _solve(Chebop(lambda u: u.diff(2) + u.sin(),
                           domain=(0.0, 5.0)))
        assert _n(u1 - u2) < TOL
