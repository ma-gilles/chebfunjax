"""Port of MATLAB Chebfun tests/unbndfun/test_mldivide.m (Opus 4.8).

``A\\B`` for array-valued unbndfuns solves the least-squares system whose
columns are the (array-valued) unbndfun columns.  chebfunjax's ``Unbndfun``
wraps a single scalar Chebtech2 and implements no ``mldivide`` / left-division;
there is no array-valued fun to solve against.

Provenance
----------
MATLAB source : tests/unbndfun/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestUnbndfunMldivide:
    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun and any mldivide "
        "(A\\B least-squares solve over quasimatrix columns)."
    )
    def test_array_valued_solve(self):
        raise NotImplementedError("array-valued Unbndfun mldivide")
