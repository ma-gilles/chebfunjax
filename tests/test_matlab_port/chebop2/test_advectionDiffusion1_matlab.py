"""Port of MATLAB Chebfun tests/chebop2/test_advectionDiffusion1.m (Opus 4.8).

Advection-diffusion ``u_t = 0.1 u_xx + u_x`` on ``[-2.5, 3] x [0, 6]`` with a
Neumann condition on the left edge and Dirichlet on the right.  The space-time
Chebop2 solution is compared against the ``chebfun/pde15s`` method-of-lines
trajectory at 61 output times in the L2 norm.

This exercises (a) a Neumann boundary condition in ``pde15s`` (imposed as an
algebraic constraint so the advection-dominated discretization stays stable)
and (b) the rectangular adaptive Chebop2 solve of a parabolic problem.

Provenance
----------
MATLAB source : tests/chebop2/test_advectionDiffusion1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


class TestChebop2Advectiondiffusion1:
    def test_all_matlab_assertions(self):
        tol = 1e10 * _EPS

        dom = (-2.5, 3.0, 0.0, 6.0)
        # Space-time Chebop2 solve: u_t - 0.1 u_xx - u_x = 0.
        N = Chebop2(
            lambda u: diffy(u, 1) - 0.1 * diffx(u, 2) - diffx(u, 1),
            domain=dom,
        )
        N.dbc = lambda x: jnp.sin(jnp.pi * x)      # initial condition at t = 0
        N.lbc = lambda t, u: u.diff()              # Neumann u_x(-2.5, t) = 0
        N.rbc = 0.0                                # Dirichlet u(3, t) = 0
        u = N.solve(0.0)

        # chebfun/pde15s reference: u_t = 0.1 u_xx + u_x.
        f = chebfun(lambda x: jnp.sin(jnp.pi * x), domain=(-2.5, 3.0))
        tt = np.arange(0.0, 6.0001, 0.1)
        UU = pde15s(
            lambda t, x, uu: 0.1 * uu.diff(2) + uu.diff(),
            tt, f, lbc=lambda uu: uu.diff(), rbc=0.0,
            n=64, rtol=1e-10, atol=1e-12,
        )

        xs = np.linspace(-2.5, 3.0, 400)
        for k, t in enumerate(tt):
            us = np.asarray(u(jnp.asarray(xs), jnp.full(xs.size, float(t))))
            ur = np.asarray(UU[k](jnp.asarray(xs)))
            err = float(np.sqrt(np.trapezoid((us - ur) ** 2, xs)))
            assert err < 10.0 * tol, f"t={t}: L2 err {err:.3e} >= {10.0 * tol:.3e}"
