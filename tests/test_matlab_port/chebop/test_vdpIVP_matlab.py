"""Port of MATLAB Chebfun tests/chebop/test_vdpIVP.m (Fable 5).

Van der Pol IVP mu=1 on [0, 10], u(0)=2, u'(0)=0, endpoint checked
against scipy's stiff integrator (MATLAB uses ode45/deval).  The
mu=1000 case is skipped (extreme stiffness; MATLAB marks it too).

Provenance
----------
MATLAB source : tests/chebop/test_vdpIVP.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp as scipy_ivp

from chebfunjax.operators.chebop import Chebop

TOL = 1e-6


class TestChebopVdpIVP:
    def test_mu1_endpoint_vs_scipy(self):
        N = Chebop(lambda x, u: u.diff(2)
                   - 1.0 * (1 - u * u) * u.diff() + u,
                   domain=(0.0, 10.0))
        N.lbc = lambda u: [u - 2.0, u.diff()]
        u = N.solve(0.0)
        uend = float(u(jnp.asarray(10.0)))
        sol = scipy_ivp(lambda t, y: [y[1],
                                      1.0 * (1 - y[0] ** 2) * y[1] - y[0]],
                        [0, 10], [2.0, 0.0], method="LSODA",
                        rtol=1e-11, atol=1e-12)
        yend = float(sol.y[0, -1])
        assert abs(uend - yend) < TOL

    def test_mu1000(self):
        pytest.skip("mu=1000 extreme-stiffness case exceeds test budget")
