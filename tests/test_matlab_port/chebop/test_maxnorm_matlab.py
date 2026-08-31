"""Port of MATLAB Chebfun tests/chebop/test_maxnorm.m (Fable 5).

pass(4)/pass(5) are bypassed in the MATLAB source itself ("This test
fails on Matlab R2015a and has been bypassed"), so only the live
assertions pass(1)-(3) and pass(6) are ported.

Provenance
----------
MATLAB source : tests/chebop/test_maxnorm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopMaxnorm:
    def test_all_matlab_assertions(self):
        T = 30.0
        t = chebfun(lambda s: s, domain=(0.0, T))
        rhs = (0.05 * t).sin()

        # %% Expect solution over whole time interval here
        N = Chebop(lambda t, y: y.diff(2) + y - 0.088 * y ** 3,
                   domain=(0.0, T))
        N.lbc = [-1.0, 0.0]
        N.maxnorm = 10.0
        y = N.solve(rhs)
        assert not y.isnan()                              # pass(1)

        # %% Blowup before we reach final time
        N = Chebop(lambda t, y: y.diff(2) + y - 0.09 * y ** 3,
                   domain=(0.0, T))
        N.lbc = [-1.0, 0.0]
        N.maxnorm = 10.0
        y = N.solve(rhs)
        assert float(y.norm(jnp.inf)) < 10.0 * 1.001      # pass(2)
        assert y.isnan()

        # %% Blowup before final time, larger maxnorm allowed
        N = Chebop(lambda t, y: y.diff(2) + y - 0.09 * y ** 3,
                   domain=(0.0, T))
        N.lbc = [-1.0, 0.0]
        N.maxnorm = 15.0
        y = N.solve(rhs)
        ynorm = float(y.norm(jnp.inf))
        assert ynorm < 15.0 * 1.001 and ynorm > 14.0      # pass(3)
        assert y.isnan()

        # %% Coupled system
        N = Chebop(lambda t, u, v: [u.diff() - 2.0 * u + u * v,
                                    v.diff() + v - u * v],
                   domain=(0.0, 10.0))
        N.lbc = lambda u, v: [u - 0.5, v - 1.0]
        N.maxnorm = [1.0, 5.0]
        sol = N.solve(0.0)
        u, v = sol[0], sol[1]
        assert (float(u.norm(jnp.inf)) < 0.9999
                or float(v.norm(jnp.inf)) < 5.0 * 0.9999)  # pass(6)
        assert u.isnan() and v.isnan()
