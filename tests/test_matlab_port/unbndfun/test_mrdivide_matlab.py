"""Port of MATLAB Chebfun tests/unbndfun/test_mrdivide.m (Opus 4.8).

``A/B`` divides an array-valued unbndfun by a scalar (case 1) or a numerical
matrix (case 2).  chebfunjax's ``Unbndfun`` wraps a single scalar Chebtech2
and has no array-valued representation, so neither the column layout nor the
matrix right-division is available.

Provenance
----------
MATLAB source : tests/unbndfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestUnbndfunMrdivide:
    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun: A/3 where A has 3 "
        "columns divides each column by the scalar; no array-valued fun."
    )
    def test_array_valued_over_scalar(self):
        raise NotImplementedError("array-valued Unbndfun / scalar")

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun and matrix mrdivide "
        "(A/B with B a 3x3 matrix)."
    )
    def test_array_valued_over_matrix(self):
        raise NotImplementedError("array-valued Unbndfun / matrix")
