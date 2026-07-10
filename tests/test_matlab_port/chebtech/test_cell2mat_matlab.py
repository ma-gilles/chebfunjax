"""Port of MATLAB Chebfun tests/chebtech/test_cell2mat.m (Opus 4.8).

MATLAB ``cell2mat([g h])`` horizontally concatenates scalar/array-valued
chebtechs into a single array-valued (quasimatrix) chebtech.  chebfunjax has no
array-valued techs and no ``cell2mat``, so this whole test is skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_cell2mat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]


class TestChebtechCell2mat:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_horizontal_concat(self, Tech, kind):
        # pass(n, 1): cell2mat([g h]) == [sin cos exp].
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs and "
            "no cell2mat (horizontal concatenation)"
        )
