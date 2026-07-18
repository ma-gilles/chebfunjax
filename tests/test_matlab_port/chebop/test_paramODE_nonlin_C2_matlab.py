"""Port of MATLAB Chebfun tests/chebop/test_paramODE_nonlin_C2.m (Fable 5).

FIXED (Fable 5, Big-Three array-valued epic): nonlinear parameter-dependent
ODEs solve as a forced square system -- the unknown scalar parameter ``a``
is carried as an extra unknown satisfying ``a' = 0``.  Here ``a`` appears in
the operator (``a*exp(u)``) and is pinned by the boundary conditions.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_nonlin_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

# MATLAB tol = 1e3 * pref.bvpTol (bvpTol default 1e-10).
TOL = 1e-7


class TestChebopParamodeNonlinC2:
    def test_nonlinear_param_ode(self):
        N = Chebop(
            lambda x, u, a: [(1 - x**2) * u + 0.1 * u.diff(2) + a * u.exp(),
                             a.diff()], (-1.0, 1.0))
        N.lbc = lambda u, a: [u + a + 1, u.diff()]
        N.rbc = lambda u, a: u - 1
        # MATLAB N.init = [x ; -1].
        N.init = [cj.chebfun(lambda x: x), cj.chebfun(lambda x: -1.0 + 0 * x)]
        sol = N.solve([0, 0], n=96)
        u, a = sol[0], sol[1]
        res = N([u, a])
        lbc = N.lbc(u, a)
        rbc = N.rbc(u, a)
        # MATLAB: err = norm(res) + Nlbc{1}(-1) + Nlbc{2}(-1) + Nrbc(1).
        err = (
            float(res[0].norm())
            + abs(float(lbc[0](jnp.asarray(-1.0))))
            + abs(float(lbc[1](jnp.asarray(-1.0))))
            + abs(float(rbc(jnp.asarray(1.0))))
        )
        assert err < TOL
        # the parameter really is constant
        xs = jnp.asarray(np.linspace(-0.99, 0.99, 40))
        assert float(jnp.max(jnp.abs(a(xs) - float(a(jnp.asarray(0.0)))))) < 1e-9
