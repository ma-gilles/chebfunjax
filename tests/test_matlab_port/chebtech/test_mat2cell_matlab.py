"""Port of MATLAB Chebfun tests/chebtech/test_mat2cell.m (Opus 4.8).

MATLAB ``mat2cell(f, 1, [1 2])`` splits an array-valued (quasimatrix) chebtech
by columns into a cell array of smaller chebtechs.  chebfunjax has no
array-valued techs and no ``mat2cell``, so both assertions are skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_SCALAR = (
    "chebfunjax Chebtech is scalar-valued; no array-valued techs and no "
    "mat2cell (column splitting)"
)


class TestChebtechMat2cell:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_first_column(self, Tech, kind):
        # pass(n, 1): F{1} == g (sin).
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_remaining_columns(self, Tech, kind):
        # pass(n, 2): F{2} == h ([cos exp]).
        pytest.skip(_SCALAR)
