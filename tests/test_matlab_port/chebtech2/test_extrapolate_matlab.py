"""Port of MATLAB Chebfun tests/chebtech2/test_extrapolate.m (Fable 5).

MATLAB's ``chebtech2.extrapolate`` replaces every sample row holding a NaN or
Inf (in any column) by the barycentric interpolant of the finite rows, at the
2nd-kind Chebyshev points; finite rows -- including the endpoints -- are
returned unchanged (they "revert").  FIXED (Fable 5): ``Chebtech2.extrapolate``
ports ``@chebtech/extrapolate.m``, both for a vector of values and for an
array-valued (2-column) block.

One convention note: chebfunjax ``chebpts(17)`` places a point at *exactly*
``x = 0`` (the odd-``n`` centre), so ``sin(x)/x`` genuinely samples ``0/0 =
NaN`` there and is legitimately filled by the interpolant.  The MATLAB
"reverting" check therefore asserts that every *finite* input row is returned
bit-for-bit unchanged (the endpoints revert), which is the property that test
exercises.

Provenance
----------
MATLAB source : tests/chebtech2/test_extrapolate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS
N = 17
_X = np.asarray(chebpts(N, 2))
_CENTER = (N - 1) // 2


def _extrap(values):
    return np.asarray(Chebtech2.extrapolate(jnp.asarray(values)))


class TestChebtech2Extrapolate:
    # -- vector-of-values iteration ------------------------------------
    def test_vector_interior_nan(self):
        # pass(1): sin(x)/x has a 0/0 NaN at the centre -> filled to ~1.
        with np.errstate(all="ignore"):
            values = np.sin(_X) / _X
        nv = _extrap(values)
        assert abs(nv[_CENTER] - 1) < TOL

    def test_vector_interior_inf(self):
        # pass(2): sin(x-eps)/x has a -Inf at the centre -> filled to ~1.
        with np.errstate(all="ignore"):
            values = np.sin(_X - EPS) / _X
        nv = _extrap(values)
        assert abs(nv[_CENTER] - 1) < TOL

    def test_vector_extrapolate_left_end(self):
        # pass(3): sign(x+1) with NaN forced at the left end -> ~1.
        values = np.sign(_X + 1).astype(float)
        values[0] = np.nan
        nv = _extrap(values)
        assert abs(nv[0] - 1) < TOL

    def test_vector_extrapolate_right_end(self):
        # pass(4): sign(x-1) with NaN forced at the right end -> ~-1.
        values = np.sign(_X - 1).astype(float)
        values[-1] = np.nan
        nv = _extrap(values)
        assert abs(nv[-1] + 1) < TOL

    def test_vector_revert_endpoints_sin(self):
        # pass(5): clean sin(x) has no NaN/Inf -> returned bit-for-bit.
        values = np.sin(_X)
        nv = _extrap(values)
        assert bool(np.all(values == nv))

    def test_vector_revert_endpoints_sinc(self):
        # pass(6): sin(x)/x -- every finite row (incl. the endpoints) reverts;
        # only the exactly-zero centre is filled (see module note).
        with np.errstate(all="ignore"):
            values = np.sin(_X) / _X
        nv = _extrap(values)
        finite = np.isfinite(values)
        assert bool(np.all(nv[finite] == values[finite]))

    # -- matrix-of-values iteration (array-valued techs) ---------------
    def test_matrix_interior_nan(self):
        with np.errstate(all="ignore"):
            col = np.sin(_X) / _X
        values = np.column_stack([col, col])
        nv = _extrap(values)
        assert np.all(np.abs(nv[_CENTER, :] - 1) < TOL)

    def test_matrix_interior_inf(self):
        with np.errstate(all="ignore"):
            col = np.sin(_X - EPS) / _X
        values = np.column_stack([col, col])
        nv = _extrap(values)
        assert np.all(np.abs(nv[_CENTER, :] - 1) < TOL)

    def test_matrix_extrapolate_left_end(self):
        col = np.sign(_X + 1).astype(float)
        values = np.column_stack([col, col])
        values[0, :] = np.nan
        nv = _extrap(values)
        assert np.all(np.abs(nv[0, :] - 1) < TOL)

    def test_matrix_extrapolate_right_end(self):
        col = np.sign(_X - 1).astype(float)
        values = np.column_stack([col, col])
        values[-1, :] = np.nan
        nv = _extrap(values)
        assert np.all(np.abs(nv[-1, :] + 1) < TOL)

    def test_matrix_revert_endpoints_sin(self):
        col = np.sin(_X)
        values = np.column_stack([col, col])
        nv = _extrap(values)
        assert bool(np.all(values == nv))

    def test_matrix_revert_endpoints_sinc(self):
        with np.errstate(all="ignore"):
            col = np.sin(_X) / _X
        values = np.column_stack([col, col])
        nv = _extrap(values)
        finite = np.isfinite(values)
        assert bool(np.all(nv[finite] == values[finite]))
