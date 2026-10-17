"""Supplemental fixed-grid parameter and boundary controls.

The16 literal original predicates, natural operators and default adaptive
solves are in test_paramODE_linearization_literal_source.py. These inherited
controls use fixed grids and analytic variants. Natural parameter problems
return numeric scalars; the explicit p.diff() system below retains a function
unknown and is a supplemental forced-system control.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_linearization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numbers

import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


def _a(x):
    """Scalar jax array for evaluating a Chebfun at a physical point."""
    return jnp.asarray(float(x))


def _deal(sol):
    """MATLAB ``[u, p] = deal(N \\ f)`` -- unpack the solution container."""
    return sol[0], sol[1]


def _bc_resid(bc_fn, u, p, domain):
    """norm of the general .bc residual (MATLAB ``norm(N.bc(x, u, p))``).

    ``bc_fn`` mirrors the general constraint callable; its outputs are the
    scalar conditions (possibly constant Chebfuns) whose 2-norm MATLAB reports.
    """
    import numpy as np

    import chebfunjax as cj
    x = cj.chebfun(lambda t: t, domain=domain)
    out = bc_fn(x, u, p)
    if not isinstance(out, (list, tuple)):
        out = [out]
    vals = []
    for c in out:
        if hasattr(c, "funs"):
            vals.append(float(c(_a(0.5 * (domain[0] + domain[1])))))
        else:
            vals.append(float(np.asarray(c).reshape(())))
    return float(np.linalg.norm(np.asarray(vals)))


class TestChebopParamodeLinearization:
    def test_supplemental_forced_second_order_field_system(self):
        # Explicit fixed16-node field system: this adds p.diff()=0 and
        # checks an actual Chebfun field parameter, separately from native
        # natural numeric parameter cases3/4 in the literal companion.
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
        # Both unknowns in this explicit differential system are Chebfuns.
        from chebfunjax.chebfun1d.chebfun import Chebfun
        assert isinstance(u, Chebfun) and isinstance(p, Chebfun)

    def test_param_only_in_bcs(self):
        # Fixed-grid variants of source cases1/2/5: the parameter appears only in the boundary
        # conditions (operator is diff(u) or diff(u,2)+u with no p).
        # FIXED (Fable 5): general .bc constraints referencing an unknown
        # numeric parameter in the natural mixed system.
        from chebfunjax.chebfun1d.chebfun import Chebfun

        # pass(1,1): op diff(u), bc = [u(-1); u(0)-c], f = exp
        A = Chebop(lambda x, u, c: u.diff(), (-1.0, 1.0))
        A.bc = lambda x, u, c: [u(_a(-1.0)), u(_a(0.0)) - c]
        import chebfunjax as cj
        f = cj.chebfun(lambda x: jnp.exp(x), domain=(-1.0, 1.0))
        u, c = _deal(A.solve(f, n=24))
        err = float((A([u, c]) - f).norm()) + _bc_resid(
            lambda x, uu, cc: [uu(_a(-1.0)), uu(_a(0.0)) - cc], u, c, (-1.0, 1.0))
        assert err < TOL
        assert isinstance(u, Chebfun) and isinstance(c, numbers.Number) and not isinstance(c, bool)

        # pass(1,2): same op, bc = u(0)-c, and an lbc u(-1)=0
        A = Chebop(lambda x, u, c: u.diff(), (-1.0, 1.0))
        A.lbc = lambda u, c: u
        A.bc = lambda x, u, c: u(_a(0.0)) - c
        u, c = _deal(A.solve(f, n=24))
        err = (
            float((A([u, c]) - f).norm())
            + _bc_resid(lambda x, uu, cc: [uu(_a(0.0)) - cc], u, c, (-1.0, 1.0))
            + abs(float(u(_a(-1.0))))
        )
        assert err < TOL
        assert isinstance(u, Chebfun) and isinstance(c, numbers.Number) and not isinstance(c, bool)

        # pass(1,5): op diff(u,2)+u (p NOT in operator), determined by the BCs
        A = Chebop(lambda x, u, p: u.diff(2) + u, (0.0, 1.0))
        A.bc = lambda x, u, p: [u(_a(0.0)), u(_a(1.0)) - 2,
                                u.diff()(_a(0.0)) - p]
        u, p = _deal(A.solve(0.0, n=24))
        err = float(A([u, p]).norm()) + _bc_resid(
            lambda x, uu, pp: [uu(_a(0.0)), uu(_a(1.0)) - 2,
                               uu.diff()(_a(0.0)) - pp], u, p, (0.0, 1.0))
        assert err < TOL
        assert isinstance(u, Chebfun) and isinstance(p, numbers.Number) and not isinstance(p, bool)

    def test_interior_point_condition(self):
        # pass(1,1)/(1,2): the condition u(0) - c is imposed at an interior
        # point of [-1, 1].
        # FIXED (Fable 5): general .bc conditions are evaluated wherever the
        # user probes u, interior points included (row-replacement in the
        # collocation system).
        from chebfunjax.chebfun1d.chebfun import Chebfun

        # A pure interior-point BVP with no parameter: u'' = -1 on [-1, 1] with
        # u(-1) = 0 and the interior condition u(0) = 1/2.  The exact solution
        # is u = 1/2 - x^2/2 (u(0) = 1/2, u(-1) = 0).
        N = Chebop(lambda x, u: u.diff(2), (-1.0, 1.0))
        N.bc = lambda x, u: [u(_a(-1.0)), u(_a(0.0)) - 0.5]
        u = N.solve(-1.0, n=24)
        assert isinstance(u, Chebfun)
        assert abs(float(u(_a(-1.0)))) < TOL
        assert abs(float(u(_a(0.0))) - 0.5) < TOL
        assert abs(float(u(_a(0.5))) - (0.5 - 0.5 * 0.25)) < TOL

        # And the MATLAB pass(1,1) case, whose u(0)-c condition is interior.
        import chebfunjax as cj
        A = Chebop(lambda x, u, c: u.diff(), (-1.0, 1.0))
        A.bc = lambda x, u, c: [u(_a(-1.0)), u(_a(0.0)) - c]
        f = cj.chebfun(lambda x: jnp.exp(x), domain=(-1.0, 1.0))
        u, c = _deal(A.solve(f, n=24))
        # c = u(0) = 1 - exp(-1) since u' = exp, u(-1) = 0
        import numpy as np
        assert abs(float(c) - (1.0 - np.exp(-1.0))) < TOL

    def test_param_determined_implicitly(self):
        # Fixed-grid variants of source cases6/7/8: the parameter is fixed only implicitly (no BC
        # references it), e.g. diff(u)-p with u(-1)=0, u(1)=2 -> p=1.
        # The natural parameter remains numeric even when the differential
        # equation determines it without a BC naming the parameter.
        from chebfunjax.chebfun1d.chebfun import Chebfun

        # pass(1,6): op diff(u)-1, lbc u=0, rbc u-p=0  ->  p = 2
        L = Chebop(lambda x, u, p: u.diff() - 1, (-1.0, 1.0))
        L.lbc = lambda u, p: u
        L.rbc = lambda u, p: u - p
        u, p = _deal(L.solve(0.0, n=24))
        err = (
            float(L([u, p]).norm())
            + abs(float(u(_a(-1.0))))
            + abs(float(u(_a(1.0))) - float(p))
        )
        assert err < TOL
        assert abs(float(p) - 2.0) < TOL
        assert isinstance(u, Chebfun) and isinstance(p, numbers.Number) and not isinstance(p, bool)

        # pass(1,7): op diff(u)-1, bc = [u(-1); u(1)-p]  ->  p = 2
        L = Chebop(lambda x, u, p: u.diff() - 1, (-1.0, 1.0))
        L.bc = lambda x, u, p: [u(_a(-1.0)), u(_a(1.0)) - p]
        u, p = _deal(L.solve(0.0, n=24))
        err = float(L([u, p]).norm()) + _bc_resid(
            lambda x, uu, pp: [uu(_a(-1.0)), uu(_a(1.0)) - pp], u, p, (-1.0, 1.0))
        assert err < TOL
        assert abs(float(p) - 2.0) < TOL

        # pass(1,8): op diff(u)-p, bc = [u(-1); u(1)-2]  ->  p = 1 (implicit)
        L = Chebop(lambda x, u, p: u.diff() - p, (-1.0, 1.0))
        L.bc = lambda x, u, p: [u(_a(-1.0)), u(_a(1.0)) - 2]
        u, p = _deal(L.solve(0.0, n=24))
        err = float(L([u, p]).norm()) + _bc_resid(
            lambda x, uu, pp: [uu(_a(-1.0)), uu(_a(1.0)) - 2], u, p, (-1.0, 1.0))
        assert err < TOL
        assert abs(float(p) - 1.0) < TOL
