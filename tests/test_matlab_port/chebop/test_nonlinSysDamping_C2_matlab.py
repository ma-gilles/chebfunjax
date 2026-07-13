"""Port of MATLAB Chebfun tests/chebop/test_nonlinSysDamping_C2.m
(Fable 5).

FIXED: second-order nonlinear coupled system with variable
coefficient and two BCs per endpoint solves via the Fable 5 block
Newton with damping.

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSysDamping_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10
D = (-np.pi, np.pi)
XS = jnp.asarray(np.linspace(-3.1, 3.1, 40))


class TestChebopNonlinSysDamping:
    def test_damped_newton_system(self):
        A = Chebop(
            lambda x, u, v: [u - v.diff(2) + (1 - x ** 2) * u ** 2,
                             u.diff(2) + v.sin()], D)
        A.lbc = lambda u, v: [u - 1, v + 1]
        A.rbc = lambda u, v: [v - 0.5, v.diff()]
        sol = A.solve([0, 0])
        u1, u2 = sol[0], sol[1]
        # pass(1): operator residual
        resid = A([u1, u2])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL
        # pass(2): boundary residuals
        a, b = D
        assert abs(float(u1(jnp.asarray(a))) - 1) < TOL
        assert abs(float(u2(jnp.asarray(a))) + 1) < TOL
        assert abs(float(u2(jnp.asarray(b))) - 0.5) < TOL
        assert abs(float(u2.diff()(jnp.asarray(b)))) < TOL
