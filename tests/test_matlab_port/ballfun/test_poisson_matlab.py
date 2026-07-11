"""Port of MATLAB Chebfun tests/ballfun/test_poisson.m (Fable 5).

Manufactured solution u = (1 - r^2) r^l Y_lm with
lap u = -(4l + 6) r^l Y_lm, u|r=1 = 0 (the identity verified in the
Opus 4.8 session; here vs the CLOSED FORM, not the library laplacian).

Provenance
----------
MATLAB source : tests/ballfun/test_poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun.spherefun import Spherefun

from ._helpers import L0, R0, T0


class TestBallfunPoisson:
    def test_manufactured_solution(self):
        ell, m = 2, 1
        Y = Spherefun.sphharm(ell, m)

        def u_exact(r, lam, th):
            return (1 - r ** 2) * r ** ell * Y(lam, th)

        def rhs(r, lam, th):
            return -(4 * ell + 6) * r ** ell * Y(lam, th)

        u = Ballfun.poisson(rhs, lmax=8, nr=24)
        got = float(u(R0, L0, T0))
        want = float(u_exact(R0, L0, T0))
        assert abs(got - want) < 1e-8
