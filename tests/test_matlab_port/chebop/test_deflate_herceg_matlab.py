"""Port of MATLAB Chebfun tests/chebop/test_deflate_herceg.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_deflate_herceg.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop, deflate


class TestChebopDeflateHerceg:
    def test_all_matlab_assertions(self):
        # Herceg equation on [0, 1]:
        #   -ep^2 u'' + (u^2 + u - 0.75)(u^2 + u - 3.75) = 0,  u(0) = u(1) = 0.
        d = (0.0, 1.0)
        ep = 0.25
        N = Chebop(
            lambda x, u: -ep**2 * u.diff(2)
            + (u * u + u - 0.75) * (u * u + u - 3.75),
            d,
        )
        N.bc = 0.0
        # x = chebfun('x', d); u0 = 0*x; N.init = u0;
        N.init = chebfun(lambda x: 0.0 * x, domain=d)

        r0 = N.solve(0.0)
        Ndef = deflate(N, r0, 1, 0)
        r1 = Ndef.solve(0.0)
        # Deflate again against both known solutions.
        Ndef = deflate(N, [r0, r1], 1, 0)
        r2 = Ndef.solve(0.0)

        # pass(1,1): all three are solutions of the original operator.
        assert float(N(r0).norm()) < 1e-9
        assert float(N(r1).norm()) < 1e-9
        assert float(N(r2).norm()) < 1e-9
        # pass(1,2): the three solutions are pairwise distinct.
        assert float((r0 - r1).norm()) > 0.1
        assert float((r0 - r2).norm()) > 0.1
        assert float((r1 - r2).norm()) > 0.1
