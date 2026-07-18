"""Port of MATLAB Chebfun tests/chebop/test_nonlinSys1Breaks_C2.m (Fable 5).

Piecewise-domain version of test_nonlinSys1_C2: a 2x2 nonlinear system
solved on ``d = [-pi 0 pi]`` with an interior breakpoint at 0.  Each
unknown is represented by two Chebyshev pieces glued with continuity
conditions at the break, so the assertions check (1) the operator
residual, (2) the boundary residuals at the endpoints, and (3) that the
solution is continuous across the breakpoint (zero jump).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSys1Breaks_C2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10
D = (-np.pi, 0.0, np.pi)
XS = jnp.asarray(np.linspace(-3.1, 3.1, 40))


def _jump(u, x0):
    """Right-limit minus left-limit of a piecewise chebfun u at x0."""
    xb = jnp.asarray(x0)
    for i in range(len(u.funs) - 1):
        if abs(float(u.funs[i].interval[1]) - float(x0)) < 1e-12:
            left = float(u.funs[i](xb))
            right = float(u.funs[i + 1](xb))
            return right - left
    return 0.0


class TestChebopNonlinsys1BreaksC2:
    def test_all_matlab_assertions(self):
        A = Chebop(
            lambda x, u, v: [u - v.diff(2) + u ** 2,
                             u.diff() + v.sin()], D)
        A.lbc = lambda u, v: u - 1
        A.rbc = lambda u, v: [v - 0.5, v.diff()]
        sol = A.solve([0, 0])
        u1, u2 = sol[0], sol[1]

        # pass(1): operator residual on a dense grid
        resid = A([u1, u2])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL

        # pass(2): boundary residuals at the endpoints
        assert abs(float(u1(jnp.asarray(D[0]))) - 1) < TOL
        assert abs(float(u2(jnp.asarray(D[-1]))) - 0.5) < TOL
        assert abs(float(u2.diff()(jnp.asarray(D[-1])))) < TOL

        # pass(3): continuous across the interior breakpoint at 0
        assert abs(_jump(u1, 0.0)) < TOL
        assert abs(_jump(u2, 0.0)) < TOL
