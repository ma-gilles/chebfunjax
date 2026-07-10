"""Port of MATLAB Chebfun tests/chebtech1/test_extrapolate.m (Opus 4.8).

MATLAB's ``chebtech1.extrapolate`` replaces interior NaN/Inf sample values
by evaluating the barycentric interpolant built from the finite samples.
chebfunjax has NO ``extrapolate`` method, so every assertion is skipped
with that precise reason.  The MATLAB file runs each check twice (once for
a vector of values, once for a 2-column matrix); the matrix repeats also
need array-valued techs, which chebfunjax lacks.

Provenance
----------
MATLAB source : tests/chebtech1/test_extrapolate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

_NO_EXTRAP = "chebfunjax has no extrapolate() (replacing interior NaN/Inf sample values)"
_NO_EXTRAP_ARRAY = (
    "chebfunjax has no extrapolate() and no array-valued/quasimatrix techs "
    "(matrix-of-values case)"
)


class TestChebtech1Extrapolate:
    # -- vector-of-values iteration ------------------------------------
    def test_vector_interior_nan(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_interior_inf(self):
        pytest.skip(_NO_EXTRAP)

    # -- matrix-of-values iteration ------------------------------------
    def test_matrix_interior_nan(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_interior_inf(self):
        pytest.skip(_NO_EXTRAP_ARRAY)
