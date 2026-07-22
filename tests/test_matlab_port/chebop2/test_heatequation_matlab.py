"""Port of MATLAB Chebfun tests/chebop2/test_heatequation.m (Fable 5).

The heat equation ``u_t = k u_xx`` with a fixed initial profile has a solution
that does not depend on the length of the time interval: solving on
``[-1,1] x [0,1]`` and on ``[-1,1] x [0,1.5]`` must agree on the common region.

Ported subset: MATLAB assertion pass(1) (interval-independence).  pass(2)-
pass(4) compare against ``chebfun/pde15s`` (a 1-D method-of-lines time
stepper) which is not part of chebfunjax, so no reference trajectory exists to
port.

Provenance
----------
MATLAB source : tests/chebop2/test_heatequation.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
