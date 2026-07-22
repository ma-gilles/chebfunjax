"""Port of MATLAB Chebfun tests/chebop/test_scalarODE_sign.m (Fable 5).

Nonlinear scalar ODE on a domain with an interior breakpoint, whose operator
carries a discontinuous coefficient ``sign(x)`` that induces a further
breakpoint in the solution at x = 0.  Solved by piecewise collocation.

Note
----
MATLAB solves the problem with three discretizations (chebcolloc2, ultraS,
chebcolloc1) and additionally checks the three solutions are not identical
(``pass(4)``); that assertion is specific to comparing distinct MATLAB
discretizations and does not apply to this single-discretization port.  The
correctness content -- a small operator residual -- is ported here.

Provenance
----------
MATLAB source : tests/chebop/test_scalarODE_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from chebfunjax.operators.chebop import Chebop  # noqa: E402

# tol = 1e3 * bvpTol, bvpTol = 5e-13.
TOL = 1e3 * 5e-13


class TestChebopScalarodeSign:
    def test_residual_small(self):
        N = Chebop(lambda x, u: u.diff(2) + x.sign() * u.sin(),
                   (-1.0, 0.5, 1.0))
        N.lbc = lambda u: u - 2
        N.rbc = lambda u: u - 2
        u1 = N.solve(0.0)
        # MATLAB pass(1): norm(N(u1)) < tol  (2-norm of the residual chebfun).
        residual = float(N(u1).norm())
        assert residual < TOL
