"""Port of MATLAB Chebfun tests/chebop/test_nonlinSysDampingBreaks_C2.m
(Fable 5).

Piecewise-domain 2x2 system on ``d = [-pi 0 pi]`` whose Newton solve needs
damping.  Each unknown is a two-piece chebfun glued with continuity at the
interior breakpoint 0.  Checks (1) the operator residual, (2) the boundary
residuals at both endpoints, and (3) continuity across the break.

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSysDampingBreaks_C2.m
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


class TestChebopNonlinsysdampingbreaksC2:
    def test_all_matlab_assertions(self):
        A = Chebop(
            lambda x, u, v: [u - v.diff(2),
                             u.diff(2) + v.cos()], D)
        A.lbc = lambda u, v: [u - 0.5, v + 0.25]
        A.rbc = lambda u, v: [v - 0.25, u + 0.5]
        sol = A.solve([0, 0])
        u1, u2 = sol[0], sol[1]

        # pass(1): operator residual on a dense grid
        resid = A([u1, u2])
        assert float(jnp.max(jnp.abs(resid[0](XS)))) < TOL
        assert float(jnp.max(jnp.abs(resid[1](XS)))) < TOL

        # pass(2): boundary residuals at both endpoints
        assert abs(float(u1(jnp.asarray(D[0]))) - 0.5) < TOL
        assert abs(float(u2(jnp.asarray(D[0]))) + 0.25) < TOL
        assert abs(float(u2(jnp.asarray(D[-1]))) - 0.25) < TOL
        assert abs(float(u1(jnp.asarray(D[-1]))) + 0.5) < TOL

        # pass(3): continuous across the interior breakpoint at 0
        assert abs(_jump(u1, 0.0)) < TOL
        assert abs(_jump(u2, 0.0)) < TOL
