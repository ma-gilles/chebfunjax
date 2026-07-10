"""Port of MATLAB Chebfun tests/chebtech2/test_extrapolate.m (Opus 4.8).

MATLAB's ``chebtech2.extrapolate`` replaces interior NaN/Inf sample values
(and, at the endpoints, NaN values) by evaluating the barycentric
interpolant built from the finite samples, and reverts endpoint values
when they are finite.  chebfunjax has NO ``extrapolate`` method, so every
assertion is skipped with that precise reason.  The MATLAB file runs each
check twice (once for a vector of values, once for a 2-column matrix); the
matrix repeats also need array-valued techs, which chebfunjax lacks.

Provenance
----------
MATLAB source : tests/chebtech2/test_extrapolate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

_NO_EXTRAP = (
    "chebfunjax has no extrapolate() (replacing interior/endpoint NaN/Inf "
    "sample values and reverting finite endpoints)"
)
_NO_EXTRAP_ARRAY = (
    "chebfunjax has no extrapolate() and no array-valued/quasimatrix techs "
    "(matrix-of-values case)"
)


class TestChebtech2Extrapolate:
    # -- vector-of-values iteration ------------------------------------
    def test_vector_interior_nan(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_interior_inf(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_extrapolate_left_end(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_extrapolate_right_end(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_revert_endpoints_sin(self):
        pytest.skip(_NO_EXTRAP)

    def test_vector_revert_endpoints_sinc(self):
        pytest.skip(_NO_EXTRAP)

    # -- matrix-of-values iteration ------------------------------------
    def test_matrix_interior_nan(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_interior_inf(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_extrapolate_left_end(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_extrapolate_right_end(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_revert_endpoints_sin(self):
        pytest.skip(_NO_EXTRAP_ARRAY)

    def test_matrix_revert_endpoints_sinc(self):
        pytest.skip(_NO_EXTRAP_ARRAY)
