"""Port of MATLAB Chebfun tests/chebop/test_manualNewton.m (Fable 5).

MATLAB's ``linearize(L, u)`` + ``mldivide(J, r)`` map to
``L.linearize(u)`` returning a solvable linearized operator with a
``solve`` method.

Provenance
----------
MATLAB source : tests/chebop/test_manualNewton.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _n(f, d=(-1.0, 1.0)):
    xs = jnp.linspace(d[0] + 1e-9, d[1] - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopManualNewton:
    def test_all_matlab_assertions(self):
        L = Chebop(lambda x, u: 0.001 * u.diff(2) - u ** 3)
        L.lbc = 1.0
        L.rbc = -1.0
        u = cj.chebfun(lambda x: -x)
        nrmdu = float("inf")
        it = 0
        while nrmdu > 1e-10 and it < 25:
            r = L * u
            J = L.linearize(u)
            du = -1.0 * J.solve(r)
            u = u + du
            nrmdu = _n(du)
            it += 1
        assert nrmdu <= 1e-10
        # sanity: the Newton fixed point satisfies the equation + BCs
        assert _n(0.001 * u.diff(2) - u ** 3) < 1e-7
        assert abs(float(u(jnp.asarray(-1.0))) - 1.0) < 1e-8
        assert abs(float(u(jnp.asarray(1.0))) + 1.0) < 1e-8
