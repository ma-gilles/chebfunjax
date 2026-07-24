"""Port of MATLAB Chebfun tests/chebop2/test_heatequation.m (Fable 5, Opus 4.8).

The heat equation ``u_t = k u_xx`` with a fixed initial profile.  ``pass(1)``
checks that the solution does not depend on the length of the time interval;
``pass(2)-pass(4)`` cross-validate the space-time Chebop2 solve against the
1-D method-of-lines time stepper ``chebfun/pde15s`` for three parameter sets.

The Chebop2 solve of a parabolic (first-order in time) problem is driven by
the rectangular adaptive resolver (many time modes, fewer space modes); the
``pde15s`` reference is run at tight tolerances so it does not dominate the
error budget.  MATLAB compares ``u(:,t)`` slices against the ``pde15s``
trajectory in the L2 norm.

Provenance
----------
MATLAB source : tests/chebop2/test_heatequation.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


def _solve_heat(tmax: float, n: int) -> "object":
    N = Chebop2(
        lambda u: diffy(u, 1) - 1.0 * diffx(u, 2),
        domain=(-1.0, 1.0, 0.0, tmax),
    )
    N.dbc = lambda x: -(x + 1.0) * (x - 1.0)  # initial profile at t = 0
    N.lbc = 0.0
    N.rbc = 0.0
    return N.solve(0.0, n=n)


def _l2_slice(u, uu_final, dom, tfinal):
    """L2_x norm of ``u(:, tfinal) - uu_final`` over the space interval."""
    xa, xb = dom
    xs = np.linspace(xa, xb, 400)
    us = np.asarray(u(jnp.asarray(xs), jnp.full(xs.size, tfinal)))
    ur = np.asarray(uu_final(jnp.asarray(xs)))
    d = us - ur
    return float(np.sqrt(np.trapezoid(d ** 2, xs)))


class TestChebop2Heatequation:
    def test_solution_independent_of_time_interval(self):
        # tol = 1e10 * eps in MATLAB, assertion uses 500 * tol.
        tol = 500.0 * 1e10 * _EPS

        u = _solve_heat(1.0, 33)
        v = _solve_heat(1.5, 33)

        xx = np.linspace(-1.0, 1.0, 40)
        yy = np.linspace(0.0, 1.0, 40)
        X, Y = np.meshgrid(xx, yy)
        uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        vv = np.asarray(v(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        assert np.max(np.abs(uv - vv)) < tol

    def _agrees_with_pde15s(self, k: float, dom4):
        """Shared body of MATLAB pass(2)-pass(4)."""
        xa, xb, ya, yb = dom4
        # chebfun/pde15s reference trajectory (final time only is compared).
        f = chebfun(lambda x: jnp.exp(-40.0 * x ** 2), domain=(xa, xb))
        UU = pde15s(
            lambda t, x, u: k * u.diff(2),
            np.arange(0.0, 1.0001, 0.1), f,
            lbc=0.0, rbc=0.0, n=64, rtol=1e-10, atol=1e-12,
        )
        # Space-time Chebop2 solve.
        N = Chebop2(lambda u: diffy(u, 1) - k * diffx(u, 2), domain=dom4)
        N.dbc = lambda x: jnp.exp(-40.0 * x ** 2)
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(0.0)
        return _l2_slice(u, UU[-1], (xa, xb), 1.0)

    def test_pass2_agrees_with_pde15s_k2(self):
        tol = 1e10 * _EPS  # MATLAB assertion: norm(...) < tol.
        err = self._agrees_with_pde15s(2.0, (-1.0, 1.0, 0.0, 1.0))
        assert err < tol

    def test_pass3_agrees_with_pde15s_kpi(self):
        tol = 1e10 * _EPS
        err = self._agrees_with_pde15s(np.pi, (-1.0, 1.0, 0.0, 1.0))
        assert err < tol

    def test_pass4_agrees_with_pde15s_wide_domain(self):
        tol = 1e10 * _EPS
        err = self._agrees_with_pde15s(1.0, (-2.0, 2.0, 0.0, 1.0))
        assert err < tol
