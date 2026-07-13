"""Port of MATLAB Chebfun tests/chebop/test_LorenzIVP.m (Fable 5).

FIXED: first-order explicit IVP systems time-march (MATLAB routes
these to ode113; chebfunjax uses LSODA with the RHS recovered by
evaluating the op on constant chebfuns and initial values solved
from the affine boundary residuals).

Provenance
----------
MATLAB source : tests/chebop/test_LorenzIVP.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.operators.chebop import Chebop

DOM = (0.0, 3.0)


class TestChebopLorenzIVP:
    def test_lorenz_vs_reference_integrator(self):
        N = Chebop(
            lambda t, u, v, w: [u.diff() - 10 * (v - u),
                                v.diff() - u * (28 - w) + v,
                                w.diff() - u * v + (8 / 3) * w],
            DOM)
        N.lbc = lambda u, v, w: [w - 20, v + 15, u + 14]
        sol = N.solve([0, 0, 0])
        u, v, w = sol[0], sol[1], sol[2]

        def ode(t, y):
            return [10 * (y[1] - y[0]),
                    y[0] * (28 - y[2]) - y[1],
                    y[0] * y[1] - (8 / 3) * y[2]]

        ref = solve_ivp(ode, DOM, [-14, -15, 20], method="LSODA",
                        rtol=1e-12, atol=1e-13, dense_output=True)
        end = ref.sol(DOM[1])
        mine = np.array([float(u(jnp.asarray(DOM[1]))),
                         float(v(jnp.asarray(DOM[1]))),
                         float(w(jnp.asarray(DOM[1])))])
        assert np.linalg.norm(mine - end) < 1e-6
