"""Port of MATLAB Chebfun tests/chebop/test_deflate_bratu.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_deflate_bratu.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.chebop import Chebop, deflate


class TestChebopDeflateBratu:
    def test_all_matlab_assertions(self):
        # Bratu equation:  u'' + 2 e^u = 0,  u(0) = u(1) = 0.
        N = Chebop(lambda x, u: u.diff(2) + 2 * u.exp(), (0.0, 1.0), 0.0, 0.0)
        r0 = N.solve(0.0)
        Ndef = deflate(N, r0, 1, 0)
        r1 = Ndef.solve(0.0)

        # pass(1,1): both are solutions of the original operator.
        assert float(N(r0).norm()) < 1e-8
        assert float(N(r1).norm()) < 1e-8
        # pass(1,2): the two solutions are distinct.
        assert float((r0 - r1).norm()) > 1.0
