"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_breakpoints.m (Fable 5).

Nonlinear scalar ODE on a domain with two interior breakpoints
(``[0 1 2 pi]``), solved by piecewise collocation.

Note
----
MATLAB solves with three discretizations and additionally checks the three
solutions differ (``pass(4)``); that assertion compares distinct MATLAB
discretizations and does not apply to this single-discretization port.  The
correctness content -- a small operator residual -- is ported here.

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_breakpoints.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

from chebfunjax.operators.chebop import Chebop  # noqa: E402

# tol = 5e4 * bvpTol, bvpTol = 5e-13.
TOL = 5e4 * 5e-13


class TestChebopScalarodeBreakpoints:
    def test_residual_small(self):
        N = Chebop(lambda x, u: u.diff(2) + (u - 0.2).sin(),
                   (0.0, 1.0, 2.0, math.pi))
        N.lbc = lambda u: u - 2
        N.rbc = lambda u: u - 3
        u1 = N.solve(0.0)
        # MATLAB pass(1): normest(N(u1)) < tol  (2-norm of the residual).
        residual = float(N(u1).norm())
        assert residual < TOL
