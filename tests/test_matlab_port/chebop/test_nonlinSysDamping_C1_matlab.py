"""Port of MATLAB Chebfun tests/chebop/test_nonlinSysDamping_C1.m (Fable 5).

The damped-Newton system under the chebcolloc1 discretization (Newton steps
solved via operators/chebop_altdisc).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSysDamping_C1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8
D = (-np.pi, np.pi)
XS = jnp.asarray(np.linspace(-3.1, 3.1, 40))


class TestChebopNonlinsysdampingC1:
    @pytest.mark.timeout(880)
    def test_all_matlab_assertions(self):
        A = Chebop(
            lambda x, u, v: [u - v.diff(2) + (1 - x ** 2) * u ** 2,
                             u.diff(2) + v.sin()], D)
        A.lbc = lambda u, v: [u - 1, v + 1]
        A.rbc = lambda u, v: [v - 0.5, v.diff()]
        sol = A.solve([0, 0], n=64, discretization="chebcolloc1")
        u1, u2 = sol[0], sol[1]
        resid = A([u1, u2])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL
        a, b = D[0], D[-1]
        assert abs(float(u1(jnp.asarray(a))) - 1) < TOL
        assert abs(float(u2(jnp.asarray(a))) + 1) < TOL
        assert abs(float(u2(jnp.asarray(b))) - 0.5) < TOL
        assert abs(float(u2.diff()(jnp.asarray(b)))) < TOL
