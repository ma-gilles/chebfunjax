"""Port of MATLAB Chebfun tests/deltafun/test_restrict.m (Opus 4.8).

chebfunjax's Deltafun has no ``restrict`` method.  MATLAB's version restricts
the funPart to each sub-interval, drops deltas outside it, and splits deltas
that land exactly on interior break points equally between adjacent pieces
(returning a cell of Deltafuns).  None of this is implemented, so every
assertion is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no restrict() method (delta splitting at "
    "break points is not implemented)"
)


class TestDeltafunRestrict:
    def test_restrict_empty(self):
        # pass(1): isempty(restrict(deltafun(), [-.5,.5]))
        pass

    def test_restrict_left_drops_delta(self):
        # pass(2): ~isa(restrict(d,[-1,-.5]), 'deltafun')
        pass

    def test_restrict_right_drops_delta(self):
        # pass(3): ~isa(restrict(d,[.5,1]), 'deltafun')
        pass

    def test_restrict_keeps_interior_delta(self):
        # pass(4): anyDelta(restrict(d,[-.5,.5]))
        pass

    def test_restrict_first_piece_locations(self):
        # pass(5): d1.deltaLoc == [-.5 -.25 0]
        pass

    def test_restrict_second_piece_domain(self):
        # pass(6): d2.funPart.domain == [0, .5]
        pass

    def test_restrict_third_piece_magnitude(self):
        # pass(7): d3.deltaMag == 1
        pass

    def test_split_first_piece(self):
        # pass(8): breakpoint delta split -> d1 loc/mag halves
        pass

    def test_split_second_piece(self):
        # pass(9): breakpoint delta split -> d2 loc/mag halves
        pass

    def test_split_third_piece(self):
        # pass(10): breakpoint delta split -> d3 loc/mag halves
        pass
