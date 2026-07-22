"""Port of MATLAB Chebfun tests/chebop/test_jumps_manual.m (Fable 5).

BVP with a nonhomogeneous interior jump condition (Toby Driscoll, based on
Chebfun issue #665): the ``.bc`` imposes ``jump(V, 0) = 1`` and a continuous
first derivative at the interior point 0, so the solution is discontinuous
there.  Solved by piecewise collocation with a breakpoint at 0.

Provenance
----------
MATLAB source : tests/chebop/test_jumps_manual.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chebfunjax.chebfun1d.chebfun import jump  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402

# tol = 1e1 * bvpTol, bvpTol = 5e-13.
TOL = 1e1 * 5e-13


class TestChebopJumpsManual:
    def test_all_matlab_assertions(self):
        N = Chebop(lambda s, V: V.diff(2), (-1.0, 1.0))
        N.lbc = lambda V: V + 1
        N.rbc = lambda V: V - 1
        N.bc = lambda s, V: [jump(V, 0.0) - 1, jump(V.diff(), 0.0)]
        # The solution is exactly piecewise linear; a low per-piece dimension
        # keeps the second-derivative residual near machine precision (the
        # single-discretization port has no adaptive dimension selection).
        y = N.solve(0.0, n=8)

        # err(1): operator residual, inf norm of N(y) = y''.
        xx = jnp.linspace(-1.0, 1.0, 401)
        err1 = float(np.max(np.abs(np.asarray(N(y)(xx)))))
        # err(2): jump in V at 0 equals 1.
        jump1 = float(y(0.0, "right")) - float(y(0.0, "left"))
        err2 = abs(jump1 - 1)
        # err(3): jump in V' at 0 is zero (continuous first derivative).
        jump2 = float(y.diff()(0.0, "right")) - float(y.diff()(0.0, "left"))
        err3 = abs(jump2)
        # err(4)/err(5): endpoint values.
        err4 = abs(float(y(-1.0)) + 1)
        err5 = abs(float(y(1.0)) - 1)

        assert err1 < TOL
        assert err2 < TOL
        assert err3 < TOL
        assert err4 < TOL
        assert err5 < TOL
