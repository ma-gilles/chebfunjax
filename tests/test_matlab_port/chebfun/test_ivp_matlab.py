"""Port of MATLAB Chebfun tests/chebfun/test_ivp.m (Fable 5).

MATLAB compares ``chebfun.odeXX`` (an ODE solve whose output is a
chebfun) against MATLAB's own ``odeXX`` at the solver's time points.
Here the reference is ``scipy.integrate.solve_ivp`` with the matching
method, evaluated at the same time points.

Provenance
----------
MATLAB source : tests/chebfun/test_ivp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from chebfunjax.chebfun1d.chebfun import ode15s, ode45, ode78, ode89, ode113

jax.config.update("jax_enable_x64", True)


def _vdp1(t, y):
    # MATLAB's vdp1: van der Pol with mu = 1.
    return jnp.array([y[1], (1.0 - y[0] ** 2) * y[1] - y[0]])


def _vdp1_np(t, y):
    return [y[1], (1.0 - y[0] ** 2) * y[1] - y[0]]


def _err(y_cols, ref):
    tm = ref.t
    vals = np.stack([np.asarray(c(jnp.asarray(tm))) for c in y_cols], axis=0)
    return float(np.max(np.abs(ref.y - vals)))


class TestChebfunIvp:
    def test_all_matlab_assertions(self):
        # pass(1): ode15s (stiff solver) on van der Pol.
        y = ode15s(_vdp1, (0.0, 5.0), jnp.array([2.0, 0.0]))
        ref = solve_ivp(_vdp1_np, [0.0, 5.0], [2.0, 0.0], method="BDF",
                        rtol=1e-3, atol=1e-6)
        assert _err(y, ref) < 2e-2

        # pass(2): ode45.
        y = ode45(_vdp1, (0.0, 5.0), jnp.array([2.0, 0.0]),
                  rtol=1e-3, atol=1e-6)
        ref = solve_ivp(_vdp1_np, [0.0, 5.0], [2.0, 0.0], method="RK45",
                        rtol=1e-3, atol=1e-6)
        assert _err(y, ref) < 1e-2

        # pass(3): ode113 with RelTol 1e-6 on [0, 20].
        y = ode113(_vdp1, (0.0, 20.0), jnp.array([2.0, 0.0]),
                   rtol=1e-6, atol=1e-6)
        ref = solve_ivp(_vdp1_np, [0.0, 20.0], [2.0, 0.0], method="DOP853",
                        rtol=1e-6, atol=1e-6)
        assert _err(y, ref) < 1e-5

        # pass(4)-(5): ode78 / ode89.
        for solver in (ode78, ode89):
            y = solver(_vdp1, (0.0, 20.0), jnp.array([2.0, 0.0]),
                       rtol=1e-6, atol=1e-6)
            assert _err(y, ref) < 1e-5

        # pass(6)-(10): a complex scalar ODE u' = i*u, u(0) = 1.
        soln = np.exp(1j)
        f = lambda t, u: 1j * u  # noqa: E731
        for solver in (ode15s, ode45, ode113, ode78, ode89):
            u = solver(f, (0.0, 1.0), jnp.array([1.0 + 0.0j]))
            assert abs(complex(u(jnp.asarray(1.0))) - soln) < 2e-2
