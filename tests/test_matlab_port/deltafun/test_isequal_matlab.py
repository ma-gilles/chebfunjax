"""Port of MATLAB Chebfun tests/deltafun/test_isequal.m (Opus 4.8).

chebfunjax's Deltafun has no ``isequal`` method (and no empty Deltafun), so
every assertion in this MATLAB test is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no isequal() method (and no empty Deltafun)"
)


class TestDeltafunIsequal:
    def test_empty_equalities(self):
        # pass(1): isequal(d1,[]) && isequal([],d1) && isequal(d1,d2) for empties
        pass

    def test_equal_identical(self):
        # pass(2): isequal(d1, d2) for identical delta data
        pass

    def test_unequal_scaled(self):
        # pass(3): ~isequal(d1, 0.992312341234*d2)
        pass

    def test_unequal_row_count(self):
        # pass(4): ~isequal(d1, d2) after dropping a magnitude row
        pass

    def test_unequal_col_count(self):
        # pass(5): ~isequal(d1, d2) after dropping a delta column/location
        pass

    def test_equal_trailing_zero_row(self):
        # pass(6): isequal([1] deltaMag, [1;0] deltaMag) (trailing zero row)
        pass

    def test_unequal_funpart(self):
        # pass(7): ~isequal(deltafun(f,[]), deltafun(bndfun([]),[]))
        pass
