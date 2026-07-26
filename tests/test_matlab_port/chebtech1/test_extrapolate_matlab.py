"""Port of MATLAB Chebfun tests/chebtech1/test_extrapolate.m (Fable 5).

MATLAB's ``chebtech1.extrapolate`` replaces every sample row holding a NaN or
Inf by the barycentric interpolant of the finite rows, at the 1st-kind
Chebyshev points.  FIXED (Fable 5): ``Chebtech1.extrapolate`` ports
``@chebtech/extrapolate.m``, both for a vector of values and for an
array-valued (2-column) block.  The interior singularity is placed at the
4th point ``x(4)`` (1-based), i.e. index 3.

Provenance
----------
MATLAB source : tests/chebtech1/test_extrapolate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS
N = 17
_X = np.asarray(chebpts(N, 1))
_IDX = 3  # MATLAB x(4), 1-based -> index 3.
_X4 = _X[_IDX]


def _extrap(values):
    return np.asarray(Chebtech1.extrapolate(jnp.asarray(values)))


class TestChebtech1Extrapolate:
    # -- vector-of-values iteration ------------------------------------
    def test_vector_interior_nan(self):
        # pass(1): sin(x-x4)/(x-x4) has a 0/0 NaN at x(4) -> filled to ~1.
        with np.errstate(all="ignore"):
            values = np.sin(_X - _X4) / (_X - _X4)
        nv = _extrap(values)
        assert abs(nv[_IDX] - 1) < TOL

    def test_vector_interior_inf(self):
        # pass(2): sin(x-x4-eps)/(x-x4) has a -Inf at x(4) -> filled to ~1.
        with np.errstate(all="ignore"):
            values = np.sin(_X - _X4 - EPS) / (_X - _X4)
        nv = _extrap(values)
        assert abs(nv[_IDX] - 1) < TOL

    # -- matrix-of-values iteration (array-valued techs) ---------------
    def test_matrix_interior_nan(self):
        with np.errstate(all="ignore"):
            col = np.sin(_X - _X4) / (_X - _X4)
        values = np.column_stack([col, col])
        nv = _extrap(values)
        assert np.all(np.abs(nv[_IDX, :] - 1) < TOL)

    def test_matrix_interior_inf(self):
        with np.errstate(all="ignore"):
            col = np.sin(_X - _X4 - EPS) / (_X - _X4)
        values = np.column_stack([col, col])
        nv = _extrap(values)
        assert np.all(np.abs(nv[_IDX, :] - 1) < TOL)
