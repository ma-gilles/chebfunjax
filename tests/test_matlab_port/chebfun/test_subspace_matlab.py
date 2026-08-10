"""Port of MATLAB Chebfun tests/chebfun/test_subspace.m (Fable 5).

MATLAB ``subspace(A, B)`` returns the largest principal angle between the column
spaces of two array-valued chebfuns.  chebfunjax exposes the equivalent
``cj.subspace(A, B)`` where ``A``/``B`` are lists of Chebfun columns
(quasimatrix form) -- so the array-valued support unblocks a direct port
(FIXED, Fable 5, Big-Three array-valued epic).

chebfunjax's Quasimatrix layer only supports SINGLE-interval domains, so we port
the test on ``[0, 2*pi]`` instead of MATLAB's piecewise ``[0 1 2*pi]``/``[0 2
2*pi]`` breakpoints -- the breakpoints are incidental (the basis
``{1/sqrt(2), cos t, sin 2t, sin 3t}/sqrt(pi)`` is orthonormal on ``[0, 2*pi]``
either way).  The row-chebfun case (MATLAB pass 6) needs a transpose, which
chebfunjax lacks, so it stays skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_subspace.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax import subspace

EPS = float(np.finfo(np.float64).eps)
PI = np.pi
SQPI = np.sqrt(PI)
DOM = (0.0, 2 * PI)


def _basis():
    # Orthonormal array-valued basis A on [0, 2*pi] (columns as a list).
    return [
        cj.chebfun(lambda t: (1.0 / np.sqrt(2.0) + 0 * t) / SQPI, domain=DOM),
        cj.chebfun(lambda t: jnp.cos(t) / SQPI, domain=DOM),
        cj.chebfun(lambda t: jnp.sin(2 * t) / SQPI, domain=DOM),
        cj.chebfun(lambda t: jnp.sin(3 * t) / SQPI, domain=DOM),
    ]


def _vscale(cols):
    return max(float(c.vscale) for c in cols)


ALPHA = [1e-10, PI / 5, PI / 2 - 1e-10]


class TestChebfunSubspace:
    @pytest.mark.parametrize("k,alpha", list(enumerate(ALPHA)))
    def test_principal_angle(self, k, alpha):
        # pass(1,2,3): B = cos(alpha)*A(:,k) + sin(alpha)*f; angle(A,B) == alpha
        # and angle(B,A) == alpha, tol 1e3*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        A = _basis()
        f = cj.chebfun(lambda t: jnp.sin(10 * t) / SQPI, domain=DOM)
        B = [float(np.cos(alpha)) * A[k] + float(np.sin(alpha)) * f]
        assert abs(subspace(A, B) - alpha) < 1e3 * EPS
        assert abs(subspace(B, A) - alpha) < 1e3 * EPS

    def test_multi_column_2_and_2(self):
        # pass(4): subspace(A(:,1:2), A(:,3:4)) == pi/2, tol 10*vscale(A)*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        A = _basis()
        assert abs(subspace(A[0:2], A[2:4]) - PI / 2) < 10 * _vscale(A) * EPS

    def test_multi_column_3_and_1(self):
        # pass(5): subspace(A(:,1:3), A(:,4)) == pi/2, tol 10*vscale(A)*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        A = _basis()
        assert abs(subspace(A[0:3], A[3:4]) - PI / 2) < 10 * _vscale(A) * EPS

    def test_row_chebfuns(self):
        # pass(6): subspace of transposed (row) chebfuns — At = A.',
        # subspace(At(1:2,:), At(3:4,:)) == pi/2.
        A = _basis()
        At = [f.transpose() for f in A]
        assert abs(subspace(At[0:2], At[2:4]) - PI / 2) \
            < 10 * _vscale(A) * EPS
