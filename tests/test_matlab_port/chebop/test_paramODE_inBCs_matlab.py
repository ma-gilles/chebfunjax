"""Supplemental fixed-grid controls related to test_paramODE_inBCs.m.

Natural parameter problems return numeric scalar parameters. These inherited
n=24 controls retain their analytic comparisons and1e-9bound; they are not a
claim of default adaptive source-test qualification. The16 original parameter
linearization predicates are in test_paramODE_linearization_literal_source.py.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_inBCs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numbers

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.chebop import Chebop

TOL = 1e-9


def _a(x):
    return jnp.asarray(float(x))


def _deal(sol):
    return sol[0], sol[1]


def _bc_resid(bc_fn, u, p, domain):
    """norm of a general .bc residual (MATLAB ``norm(bc(x, u, p))``)."""
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


class TestChebopParamodeInbcs:
    def test_second_order_param_in_bc(self):
        # Second order problem with the parameter in a BC:
        #   diff(u,2) + u = 0 on [0, 1],
        #   u(0) = 0, u(1) = 2, u'(0) = p.
        dom = (0.0, 1.0)
        bc = lambda x, u, p: [u(_a(0.0)), u(_a(1.0)) - 2,
                              u.diff()(_a(0.0)) - p]
        A = Chebop(lambda x, u, p: u.diff(2) + u, dom)
        A.bc = bc
        u, p = _deal(A.solve(0.0, n=24))
        err = float(A([u, p]).norm()) + _bc_resid(bc, u, p, dom)
        assert err < TOL
        assert isinstance(u, Chebfun) and isinstance(p, numbers.Number) and not isinstance(p, bool)

    def test_example1_lbc_rbc(self):
        # L = diff(u) - 1 on [-1, 1], lbc u = 0, rbc u - p = 0.
        # Solution: p = 2, u the straight line from (-1, 0) to (1, 2).
        dom = (-1.0, 1.0)
        L = Chebop(lambda t, u, p: u.diff() - 1, dom)
        L.lbc = lambda u, p: u
        L.rbc = lambda u, p: u - p
        u, p = _deal(L.solve(0.0, n=24))
        line = cj.chebfun(lambda t: 1.0 + t, domain=dom)
        err = float((u - line).norm()) + abs(float(p) - 2.0)
        assert err < TOL

    def test_example2_bc_field(self):
        # Same as example 1, but conditions via the general .bc field.
        dom = (-1.0, 1.0)
        L = Chebop(lambda t, u, p: u.diff() - 1, dom)
        L.bc = lambda t, u, p: [u(_a(-1.0)), u(_a(1.0)) - p]
        u, p = _deal(L.solve(0.0, n=24))
        line = cj.chebfun(lambda t: 1.0 + t, domain=dom)
        err = float((u - line).norm()) + abs(float(p) - 2.0)
        assert err < TOL

    def test_example3_param_in_operator(self):
        # Parameter appears in the differential equation, not the BCs:
        #   diff(u) - p = 0, u(-1) = 0, u(1) = 2  ->  p = 1 (implicit).
        dom = (-1.0, 1.0)
        L = Chebop(lambda t, u, p: u.diff() - p, dom)
        L.bc = lambda t, u, p: [u(_a(-1.0)), u(_a(1.0)) - 2]
        u, p = _deal(L.solve(0.0, n=24))
        line = cj.chebfun(lambda t: 1.0 + t, domain=dom)
        err = float((u - line).norm()) + abs(float(p) - 1.0)
        assert err < TOL
