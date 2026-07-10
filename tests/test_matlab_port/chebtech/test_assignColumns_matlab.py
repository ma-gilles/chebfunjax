"""Port of MATLAB Chebfun tests/chebtech/test_assignColumns.m (Opus 4.8).

MATLAB ``assignColumns(f, cols, g)`` overwrites selected columns of an
array-valued (quasimatrix) chebtech.  chebfunjax has no array-valued techs and
no ``assignColumns``, so all three assertions are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_assignColumns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_SCALAR = (
    "chebfunjax Chebtech is scalar-valued; no array-valued techs and no "
    "assignColumns (column assignment)"
)


class TestChebtechAssignColumns:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_assign_two_columns(self, Tech, kind):
        # pass(1): assignColumns(f, [1 3], g) == [x cos(x) x^2].
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_assign_unhappy_column(self, Tech, kind):
        # pass(2): assigning sqrt(x) -> ~ishappy.
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_delete_column(self, Tech, kind):
        # pass(3): assignColumns(f, 1, []) -> size(vscale) == [1 2].
        pytest.skip(_SCALAR)
