"""Port of MATLAB Chebfun tests/chebop/test_paramODE_nonlin_US.m (Fable 5).

The parameter-dependent system under the ultraS discretization (Newton steps
solved via operators/chebop_altdisc).

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_nonlin_US.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8
import chebfunjax as cj  # noqa: E402


class TestChebopParamodeNonlinUS:
    @pytest.mark.timeout(880)
    def test_all_matlab_assertions(self):
        N = Chebop(
            lambda x, u, a: [(1 - x**2) * u + 0.1 * u.diff(2)
                             + a * u.exp(), a.diff()], (-1.0, 1.0))
        N.lbc = lambda u, a: [u + a + 1, u.diff()]
        N.rbc = lambda u, a: u - 1
        x = cj.chebfun(lambda t: t)
        N.init = [x, cj.chebfun(lambda t: -1.0 + 0 * t)]
        sol = N.solve([0, 0], n=96, discretization="ultraS")
        u, a = sol[0], sol[1]
        res = N([u, a])
        lbc = N.lbc(u, a)
        rbc = N.rbc(u, a)
        err = (float(res[0].norm())
               + abs(float(lbc[0](jnp.asarray(-1.0))))
               + abs(float(lbc[1](jnp.asarray(-1.0))))
               + abs(float(rbc(jnp.asarray(1.0)))))
        assert err < 1e-7
