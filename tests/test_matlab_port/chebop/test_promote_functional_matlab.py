"""Port of MATLAB Chebfun tests/chebop/test_promote_functional.m
(Fable 5).

Integro-differential operators (sum(u) terms inside the op).  The
nonlinear Newton path resolves them to 1e-13; the scalar LINEAR
path's functional promotion only reaches ~7e-9 against MATLAB's
1e-10 bound (worse at higher n), so that assertion is an honest
xfail with evidence.  The system case (pointwise u(0)-style bc)
remains a documented gap.

Provenance
----------
MATLAB source : tests/chebop/test_promote_functional.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10
XS = jnp.asarray(np.linspace(-0.99, 0.99, 30))


class TestChebopPromoteFunctional:
    # (Previously xfailed at ~7e-9: the solve returned an unchopped
    # ~1e-14 coefficient tail that diff(2) amplified in the residual;
    # solutions are now simplified as in MATLAB's linsolve.  Measured
    # 6.4e-14 on 2026-07-30.)
    def test_linear_integro_differential(self):
        # pass(1)
        N = Chebop(lambda x, u: u.diff(2) + u.sum(), (-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(1.0)
        assert float(jnp.max(jnp.abs(N(u)(XS) - 1.0))) < TOL

    def test_nonlinear_integro_differential(self):
        # pass(2)
        N = Chebop(lambda x, u: u.diff(2) + u * u.sum(),
                   (-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(1.0)
        assert float(jnp.max(jnp.abs(N(u)(XS) - 1.0))) < TOL
