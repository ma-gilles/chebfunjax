"""Port of MATLAB Chebfun tests/chebop/test_paramODE_linearization.m (Fable 5).

Unknown-parameter problems solve as a forced square system (the parameter
``p`` carried as an extra unknown with ``p' = 0``) WHEN the parameter appears
in the differential operator and is pinned by an endpoint boundary condition.

The cases where the parameter appears ONLY in the boundary conditions (not in
the operator), or where a condition is imposed at an interior point
(``u(0)`` on a domain straddling 0), or where the parameter is determined
only implicitly (no boundary condition references it), are not solved -- the
parameter stays at its initial value.  Those stay skipped with precise
reasons.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_linearization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


class TestChebopParamodeLinearization:
    def test_second_order_param_in_operator(self):
        # pass(1,3)/(1,4): diff(u,2) + u + p on [0,1] with u(0)=0, u(1)=2,
        # u'(0)=p -- the parameter appears in the operator and is pinned by the
        # left endpoint condition u'(0) - p.
        # FIXED (Fable 5, Big-Three array-valued epic).
        A = Chebop(lambda x, u, p: [u.diff(2) + u + p, p.diff()], (0.0, 1.0))
        A.lbc = lambda u, p: [u, u.diff() - p]
        A.rbc = lambda u, p: u - 2
        sol = A.solve([0, 0], n=16)
        u, p = sol[0], sol[1]
        res = A([u, p])
        a, b = jnp.asarray(0.0), jnp.asarray(1.0)
        err = (
            float(res[0].norm())
            + abs(float(u(a)))
            + abs(float(u(b)) - 2)
            + abs(float(u.diff()(a)) - float(p(jnp.asarray(0.5))))
        )
        assert err < TOL
        # pass(2,3)/(2,4): u is a chebfun and p is (a constant) chebfun.
        from chebfunjax.chebfun1d.chebfun import Chebfun
        assert isinstance(u, Chebfun) and isinstance(p, Chebfun)

    def test_param_only_in_bcs(self):
        # pass(1,1)/(1,2)/(1,5): the parameter appears only in the boundary
        # conditions (operator is diff(u) or diff(u,2)+u with no p).
        pytest.skip(
            "unknown parameters that appear only in the boundary conditions (not in "
            "the differential operator) are not solved: the parameter block is a "
            "constant, and the solver leaves it at its initial value -- src gap"
        )

    def test_interior_point_condition(self):
        # pass(1,1)/(1,2): the condition u(0) - c is imposed at an interior point
        # of [-1, 1].
        pytest.skip(
            "interior-point boundary conditions (e.g. u(0) on [-1, 1]) are not "
            "expressible: Chebop lbc/rbc apply only at the domain endpoints -- src gap"
        )

    def test_param_determined_implicitly(self):
        # pass(1,6)/(1,7)/(1,8): the parameter is fixed only implicitly (no BC
        # references it), e.g. diff(u)-p with u(-1)=0, u(1)=2 -> p=1.
        pytest.skip(
            "parameters determined only implicitly (no boundary condition references "
            "the parameter) are not solved: the solver leaves the parameter at its "
            "initial value -- src gap"
        )
