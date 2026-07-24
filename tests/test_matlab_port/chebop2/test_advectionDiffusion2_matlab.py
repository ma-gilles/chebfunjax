"""Port of MATLAB Chebfun tests/chebop2/test_advectionDiffusion2.m (Opus 4.8).

Advection-diffusion ``u_t = 0.3 u_xx + 10 u_x`` on ``[-1, 1] x [0, 0.25]`` with
homogeneous Dirichlet conditions and a sharp bump initial condition.  The
space-time Chebop2 solution is compared against the ``chebfun/pde15s``
method-of-lines trajectory at 51 output times in the L2 norm.

Provenance
----------
MATLAB source : tests/chebop2/test_advectionDiffusion2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


def _ic(x):
    # exp(-10 x^4 / (1 - x^2)); vanishes (with all derivatives) at x = +/-1.
    return jnp.exp(-10.0 * x ** 4 / (1.0 - x ** 2))


class TestChebop2Advectiondiffusion2:
    def test_all_matlab_assertions(self):
        tol = 1e10 * _EPS

        dom = (-1.0, 1.0, 0.0, 0.25)
        # Space-time Chebop2 solve: u_t - 0.3 u_xx - 10 u_x = 0.
        N = Chebop2(
            lambda u: diffy(u, 1) - 0.3 * diffx(u, 2) - 10.0 * diffx(u, 1),
            domain=dom,
        )
        N.dbc = _ic
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(0.0)

        # chebfun/pde15s reference: u_t = 0.3 u_xx + 10 u_x.
        f = chebfun(_ic, domain=(-1.0, 1.0))
        tt = np.arange(0.0, 0.250001, 0.005)
        UU = pde15s(
            lambda t, x, uu: 0.3 * uu.diff(2) + 10.0 * uu.diff(),
            tt, f, lbc=0.0, rbc=0.0, n=80, rtol=1e-10, atol=1e-12,
        )

        xs = np.linspace(-1.0, 1.0, 400)
        for k, t in enumerate(tt):
            us = np.asarray(u(jnp.asarray(xs), jnp.full(xs.size, float(t))))
            ur = np.asarray(UU[k](jnp.asarray(xs)))
            err = float(np.sqrt(np.trapezoid((us - ur) ** 2, xs)))
            assert err < 2.0 * tol, f"t={t}: L2 err {err:.3e} >= {2.0 * tol:.3e}"
