"""Port of MATLAB Chebfun tests/chebop/test_paramODE.m (Fable 5).

FIXED (forced-system block): parameter-dependent ODEs solve as
square systems with the parameter as an unknown satisfying a' = 0
(the MATLAB test's own 'forced setup using a system').  The
rectangular 'natural setup' (one equation, two unknowns with the
parameter inferred) remains a documented gap.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e-8
XS = jnp.asarray(np.linspace(-0.99, 0.99, 40))


class TestChebopParamODE:
    def test_forced_system_setup(self):
        N = Chebop(
            lambda x, u, a: [x * u + 0.001 * u.diff(2) + a,
                             a.diff()], (-1.0, 1.0))
        N.lbc = lambda u, a: [u + a + 1, u.diff()]
        N.rbc = lambda u, a: u - 1
        sol = N.solve([0, 0], n=96)
        u, a = sol[0], sol[1]
        resid = N([u, a])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL
        # the parameter really is constant
        assert float(jnp.max(jnp.abs(
            a(XS) - float(a(jnp.asarray(0.0)))))) < 1e-10
        # boundary residuals
        assert abs(float(u(jnp.asarray(-1.0)))
                   + float(a(jnp.asarray(-1.0))) + 1) < TOL
        assert abs(float(u.diff()(jnp.asarray(-1.0)))) < TOL
        assert abs(float(u(jnp.asarray(1.0))) - 1) < TOL
