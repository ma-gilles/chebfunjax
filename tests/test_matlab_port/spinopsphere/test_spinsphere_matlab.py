"""Port of MATLAB Chebfun tests/spinopsphere/test_spinsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinopsphere/test_spinsphere.m
Chebfun commit: 7574c77

The MATLAB test is a self-convergence (Cauchy) check: solve the PDE with
time-step DT and DT/2 at N=128 grid points, then compare the two
solutions on a 50x50 grid.  MATLAB tolerance ``tol = 1e-2``.

Measured self-convergence errors (this port, deterministic):
    AC: 0.0012   (~8x margin)
    GL: 0.0089   (the genuine 4th-order IMEX-BDF4 error at dt=0.1;
                  the convergence ratio dt/2 -> dt/4 is ~16, confirming
                  clean fourth order -- this is the same error MATLAB
                  incurs, hence the shared 1e-2 threshold is kept as-is).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.spinopsphere import Spinopsphere, spinsphere
from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-2  # MATLAB test tolerance (do not widen)


def _compare(u, v):
    """max|u - v| / max|v| on the MATLAB 50x50 grid over [-pi, pi]^2."""
    lam = np.linspace(-np.pi, np.pi, 50)
    xx, yy = np.meshgrid(lam, lam)
    uu = np.asarray(u(jnp.asarray(xx), jnp.asarray(yy)))
    vv = np.asarray(v(jnp.asarray(xx), jnp.asarray(yy)))
    scale = float(np.max(np.abs(vv)))
    return float(np.max(np.abs(uu - vv))) / scale


class TestSpinopsphereSpinsphere:
    def test_all_matlab_assertions(self):
        # ---- AC ------------------------------------------------------
        #   S = spinopsphere('AC'); S.tspan = S.tspan/10;
        #   N = 128; dt = 1e-1;
        #   u = spinsphere(S, N, dt, 'plot', 'off');
        #   v = spinsphere(S, N, dt/2, 'plot', 'off');
        S = Spinopsphere("AC")
        S.tspan = tuple(t / 10 for t in S.tspan)
        N = 128
        dt = 1e-1
        u = spinsphere(S, N, dt, "plot", "off")
        v = spinsphere(S, N, dt / 2, "plot", "off")
        err_ac = _compare(u, v)
        assert err_ac < TOL, f"AC self-convergence {err_ac} !< {TOL}"

        # ---- GL ------------------------------------------------------
        #   S = spinopsphere('GL'); S.tspan = S.tspan/10;
        #   S.init = spherefun(@(x,y,z) cos(cosh(x.*z)-y));
        #   N = 128; dt = 1e-1;
        S = Spinopsphere("GL")
        S.tspan = tuple(t / 10 for t in S.tspan)

        def gl_init(lam, th):
            x = jnp.sin(th) * jnp.cos(lam)
            y = jnp.sin(th) * jnp.sin(lam)
            z = jnp.cos(th)
            return jnp.cos(jnp.cosh(x * z) - y)

        S.init = Spherefun.from_function(gl_init)
        u = spinsphere(S, N, dt, "plot", "off")
        v = spinsphere(S, N, dt / 2, "plot", "off")
        err_gl = _compare(u, v)
        assert err_gl < TOL, f"GL self-convergence {err_gl} !< {TOL}"
