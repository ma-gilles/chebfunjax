"""Native Van der Pol endpoint predicates, Chebfun commit: 7574c77.

MATLAB source: tests/chebop/test_vdpIVP.m, both mu=1 and mu=1000.
The native endpoint oracle uses ode45 with AbsTol=1e5*eps and RelTol=100*eps.
Captured MATLAB R2025b endpoints replace that external reference solver;
source Chebop defaults and the strict 1e-10 bound are unchanged. Capture:
vdp_native_capture_v2_20261011/observations.mat, SHA256
54f5618452f78dcfd7bd935aca9b8635b229f805c632c306e8b13ffca5790f3f.
The historical SciPy comparison remains a separate supplemental check.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp as scipy_ivp

from chebfunjax.operators.chebop import Chebop

TOL = 1e-6


class TestChebopVdpIVP:
    def test_mu1_endpoint_vs_scipy(self):
        N = Chebop(lambda x, u: u.diff(2) - 1.0 * (1 - u * u) * u.diff() + u, domain=(0.0, 10.0))
        N.lbc = lambda u: [u - 2.0, u.diff()]
        u = N.solve(0.0)
        uend = float(u(jnp.asarray(10.0)))
        sol = scipy_ivp(
            lambda t, y: [y[1], 1.0 * (1 - y[0] ** 2) * y[1] - y[0]],
            [0, 10],
            [2.0, 0.0],
            method="LSODA",
            rtol=1e-11,
            atol=1e-12,
        )
        yend = float(sol.y[0, -1])
        assert abs(uend - yend) < TOL


@pytest.mark.parametrize(
    ("mu", "native_ode45_endpoint"),
    [(1.0, -2.0083407825769455), (1000.0, 1.9933149275697708)],
)
def test_native_endpoint(mu, native_ode45_endpoint):
    """Both source err < 1e-10 predicates, with omitted solver overrides."""
    N = Chebop(lambda x, u: u.diff(2) - mu * (1 - u * u) * u.diff() + u, domain=(0.0, 10.0))
    N.lbc = lambda u: [u - 2.0, u.diff()]
    u = N.solve(0.0)
    endpoint = float(u(jnp.asarray(10.0)))
    assert abs(endpoint - native_ode45_endpoint) < 1e-10
