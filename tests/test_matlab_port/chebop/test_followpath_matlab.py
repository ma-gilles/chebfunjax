"""Port of MATLAB Chebfun tests/chebop/test_followpath.m (Fable 5).

``followpath(N, lam0, 'name', val, ...)`` maps to keyword arguments of
:func:`chebfunjax.operators.followpath.followpath`.

Provenance
----------
MATLAB source : tests/chebop/test_followpath.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.followpath import followpath

jax.config.update("jax_enable_x64", True)


class TestChebopFollowpath:
    def test_all_matlab_assertions(self):
        # pass(1)/(2): Bratu problem u'' + lam*exp(u) = 0.
        N = Chebop(lambda x, u, lam: u.diff(2) + lam * u.exp(),
                   domain=(0.0, 1.0))
        N.lbc = lambda u, lam: u
        N.rbc = lambda u, lam: u
        lam0 = 0.01
        u, lamvec, *_ = followpath(N, lam0, maxstepno=6)
        assert len(lamvec) >= 2                                     # pass(1)
        u, lamvec, mvec, *_ = followpath(
            N, lam0, measure=lambda u: float(u(jnp.asarray(0.5))),
            maxstepno=5)
        assert len(mvec) == len(lamvec)                             # pass(2)

        # pass(3): Allen-Cahn-type problem with lam entering the BC.
        d = (0.0, 1.0)
        ep = 2.0 ** -2
        N0 = Chebop(lambda x, u: -ep ** 2 * u.diff(2)
                    + (u ** 2 + u - 0.75) * (u ** 2 + u - 3.75), domain=d)
        N0.lbc = 0.0
        N0.rbc = 0.0
        u0 = N0.solve(0.0)
        N = Chebop(lambda x, u, lam: -ep ** 2 * u.diff(2)
                   + (u ** 2 + u - 0.75) * (u ** 2 + u - 3.75), domain=d)
        lam0 = float(u0.diff()(jnp.asarray(d[1])))
        N.lbc = lambda u, lam: u
        N.rbc = lambda u, lam: u.diff() - lam
        u, lamvec, mvec, lamfun, mfun = followpath(
            N, lam0, maxstepno=6, uinit=u0,
            measure=lambda u: float(u(jnp.asarray(1.0))), stepmax=0.1)
        assert len(lamvec) >= 2                                     # pass(3)

        # pass(4): lam multiplying the highest derivative.
        H = Chebop(lambda x, u, lam: -lam * u.diff(2)
                   + (u ** 2 + u - 0.75) * (u ** 2 + u - 3.75),
                   domain=(0.0, 1.0))
        lam0 = ep ** 2
        H.lbc = lambda u, lam: u
        H.rbc = lambda u, lam: u
        u, lamvec, mvec, *_ = followpath(
            H, lam0, measure=lambda u: float(u.diff().norm(jnp.inf)),
            direction=-1, stopfun=lambda u, lam: lam < 2e-2,
            stepmax=0.1)
        assert len(lamvec) >= 2                                     # pass(4)
