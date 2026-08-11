"""Port of MATLAB Chebfun tests/chebfun2/test_chebpolyval2.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``chebpolyval2`` gained the
low-rank three-factor form, and the static helpers
``Chebfun2.chebpts2`` / ``Chebfun2.coeffs2vals`` were added.

Provenance
----------
MATLAB source : tests/chebfun2/test_chebpolyval2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.polynomials import chebeval

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Chebpolyval2:
    def test_coeffs2vals_of_single_coefficient(self):
        # pass(1): the values of T_n(x) T_n(y) on an N-by-N grid equal
        # coeffs2vals of the unit coefficient matrix e_{n,n}.
        n = 20
        N = 5 * n
        xx, yy = Chebfun2.chebpts2(N, N)
        A = np.asarray(chebeval(xx, n) * chebeval(yy, n))
        C = np.zeros((N, N))
        C[n, n] = 1.0
        X = np.asarray(Chebfun2.coeffs2vals(C))
        assert float(np.linalg.norm(A - X)) < 100 * TOL

    def test_low_rank_factors_multiply_back(self):
        # pass(2): X == A1 * A2 * A3.'
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(x * y) + jnp.sin(x) + jnp.exp(y))
        A1, A2, A3 = f.chebpolyval2(low_rank=True)
        X = np.asarray(f.chebpolyval2())
        prod = np.asarray(A1) @ np.asarray(A2) @ np.asarray(A3).T
        assert float(np.linalg.norm(X - prod)) < TOL

    def test_recovers_values_matrix(self):
        # pass(3): chebpolyval2(chebfun2(A)) recovers A, including when
        # the x- and y-degrees differ.
        rng = np.random.default_rng(0)
        A = rng.random((3, 4))
        f = Chebfun2.from_values(jnp.asarray(A))
        B = np.asarray(f.chebpolyval2())
        assert B.shape == A.shape
        assert float(np.linalg.norm(A - B)) < TOL
