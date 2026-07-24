"""Port of MATLAB Chebfun tests/chebop/test_deflate_painleve.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_deflate_painleve.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

from chebfunjax.operators.chebop import Chebop, deflate


class TestChebopDeflatePainleve:
    def test_all_matlab_assertions(self):
        # Painleve I:  u'' - u^2 + x = 0,  u(0) = 0,  u(L) = sqrt(L).
        L = 10.0
        N = Chebop(lambda x, u: u.diff(2) - u * u + x, (0.0, L))
        N.lbc = 0.0
        N.rbc = math.sqrt(L)

        r0 = N.solve(0.0)  # first solution
        Ndef = deflate(N, r0, 3, 0.1)  # deflate for a second solution
        r1 = Ndef.solve(0.0)

        # pass(1,1): both are solutions of the original operator.
        assert float(N(r0).norm()) < 1e-9
        assert float(N(r1).norm()) < 1e-9
        # pass(1,2): the two solutions are distinct.
        assert float((r0 - r1).norm()) > 1.0
