"""Port of MATLAB Chebfun tests/deltafun/test_zeroDeltaFun.m (Fable 5).

FIXED: Deltafun.zero_delta_fun added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/deltafun/test_zeroDeltaFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.domain import Domain
from chebfunjax.fun.deltafun import Deltafun


class TestDeltafunZeroDeltaFun:
    def test_zero_everywhere(self):
        z = Deltafun.zero_delta_fun()
        assert z.iszero()
        assert z.n_deltas == 0
        assert abs(float(z.sum())) < 1e-15

    def test_custom_domain(self):
        z = Deltafun.zero_delta_fun(Domain((0.0, 5.0)))
        assert z.iszero()
        assert float(z.domain.a) == 0.0 and float(z.domain.b) == 5.0
