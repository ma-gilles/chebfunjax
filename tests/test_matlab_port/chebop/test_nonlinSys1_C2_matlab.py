"""Port of MATLAB Chebfun tests/chebop/test_nonlinSys1_C2.m
(Fable 5).

FIXED: nonlinear coupled systems with mixed boundary-condition
counts solve via the Fable 5 block Newton (lbc one residual, rbc
two including a derivative).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSys1_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10
D = (-np.pi, np.pi)
XS = jnp.asarray(np.linspace(-3.1, 3.1, 40))


class TestChebopNonlinSys1:
    def test_mixed_bc_nonlinear_system(self):
        A = Chebop(
            lambda x, u, v: [u - v.diff(2) + u ** 2,
                             u.diff() + v.sin()], D)
        A.lbc = lambda u, v: u - 1
        A.rbc = lambda u, v: [v - 0.5, v.diff()]
        sol = A.solve([0, 0])
        u1, u2 = sol[0], sol[1]
        # pass(1): operator residual
        resid = A([u1, u2])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL
        # pass(2): boundary residuals
        assert abs(float(u1(jnp.asarray(D[0]))) - 1) < TOL
        assert abs(float(u2(jnp.asarray(D[1]))) - 0.5) < TOL
        assert abs(float(u2.diff()(jnp.asarray(D[1])))) < TOL
