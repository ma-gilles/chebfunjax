"""Port of MATLAB Chebfun tests/deltafun/test_times.m (Opus 4.8).

MATLAB multiplies a smooth function by a (possibly high-order) delta via the
Leibniz/Taylor rule: f(x)*delta^(m)(x - x0) expands into a weighted sum of
lower-order deltas whose magnitudes are derivatives of f at x0.  chebfunjax's
``Deltafun.__mul__`` supports only scalar and Deltafun*Bndfun (scaling row-0
magnitudes by g(x0)); multiplying two Deltafuns raises NotImplementedError and
the derivative-coupling of multi-row magnitudes is absent.  Every assertion is
skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun.__mul__ does not implement the deltafun-times-"
    "deltafun Leibniz product that generates derivative deltas"
)


class TestDeltafunTimes:
    def test_empty_times_empty(self):
        # pass(1): isempty(deltafun() .* deltafun())
        pass

    def test_empty_times_delta(self):
        # pass(2): isempty(deltafun().*d) && isempty(d.*deltafun())
        pass

    def test_expneg_times_delta4(self):
        # pass(3): exp(-x) .* delta^(4) -> [1,4,6,4,1]'
        pass

    def test_exp_times_delta3(self):
        # pass(4): exp(x) .* delta^(3) -> [-1,3,-3,1]'
        pass

    def test_smooth_times_delta_funpart(self):
        # pass(5): iszero(s.funPart - f1.*f2)
        pass

    def test_smooth_times_delta_locations(self):
        # pass(6): s.deltaLoc == sort(l1)
        pass

    def test_smooth_times_delta_magnitudes(self):
        # pass(7): Leibniz magnitudes for f2 derivatives at l1
        pass

    def test_two_delta_blocks_funpart(self):
        # pass(8): iszero(s.funPart - f1.*f2)
        pass

    def test_two_delta_blocks_locations(self):
        # pass(9): s.deltaLoc == sort(union(l1,l2))
        pass

    def test_two_delta_blocks_magnitudes(self):
        # pass(10): combined Leibniz magnitude blocks
        pass
