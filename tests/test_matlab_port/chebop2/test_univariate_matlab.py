"""Port of MATLAB Chebfun tests/chebop2/test_univariate.m (Opus 4.8).

Rank-1 PDEs (a differential operator in a single variable) solved with the
coefficient-space Chebop2 path.  The wrapped solution must have numerical rank
1, and its slice must match the corresponding 1-D solution.

The 1-D reference ``u'' + u = 0`` with ``u(-1) = u(1) = 1`` is the entire
function ``cos(x)/cos(1)`` -- the same solution the MATLAB test computes with a
1-D chebop.

Provenance
----------
MATLAB source : tests/chebop2/test_univariate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2

_EPS = float(np.finfo(np.float64).eps)


def _ev(u, x, y):
    return np.asarray(u(jnp.asarray(np.atleast_1d(x), dtype=jnp.float64),
                        jnp.asarray(np.atleast_1d(y), dtype=jnp.float64)))


class TestChebop2Univariate:
    def test_all_matlab_assertions(self):
        tol = 1000.0 * _EPS  # MATLAB 1000*cheb2Prefs.chebfun2eps.
        ref = lambda s: np.cos(s) / np.cos(1.0)

        # --- y-variable example: u_yy + u = 0, u(y=-1)=u(y=1)=1 ---
        N = Chebop2(lambda u: u.diff(2, 0) + u)
        N.dbc = 1.0
        N.ubc = 1.0
        u = N.solve(0.0)
        pivots = np.asarray(u.pivots)
        rk = int(np.sum(np.abs(pivots) / max(abs(pivots[0]), 1e-300) > 1e-12))
        assert rk == 1  # pass(1): length(u) == 1.
        ys = np.linspace(-1.0, 1.0, 40)
        assert np.max(np.abs(_ev(u, np.zeros_like(ys), ys) - ref(ys))) < tol  # pass(2).

        # --- x-variable example: u_xx + u = 0, u(x=-1)=u(x=1)=1 ---
        N = Chebop2(lambda u: u.diff(0, 2) + u)
        N.lbc = 1.0
        N.rbc = 1.0
        u = N.solve(0.0)
        pivots = np.asarray(u.pivots)
        rk = int(np.sum(np.abs(pivots) / max(abs(pivots[0]), 1e-300) > 1e-12))
        assert rk == 1  # pass(3).
        xs = np.linspace(-1.0, 1.0, 40)
        assert np.max(np.abs(_ev(u, xs, np.zeros_like(xs)) - ref(xs))) < tol  # pass(4).
