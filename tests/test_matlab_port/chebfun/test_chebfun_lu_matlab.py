"""Port of MATLAB Chebfun tests/chebfun/test_chebfun_lu.m (Fable 5).

MATLAB's ``norm`` of a quasimatrix defaults to the Frobenius norm, and
``L(p, :)`` is the matrix of the columns of ``L`` evaluated at the pivot
locations.

Provenance
----------
MATLAB source : tests/chebfun/test_chebfun_lu.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix, lu
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _fro(Q):
    return float(np.sqrt(sum(float(c.norm(2)) ** 2 for c in Q.cols)))


def _check(x, tol, res_tol, tril_tol):
    A = Quasimatrix([0 * x + 1, x, x ** 2, x ** 3, x ** 4, x ** 5],
                    x.domain)
    L, U, p = lu(A)
    U = np.asarray(U)
    assert np.linalg.norm(np.triu(U) - U, 2) < tol                     # triu(U) == U
    LU = Quasimatrix([sum((L.cols[i] * float(U[i, j]) for i in range(6)),
                          0 * x) for j in range(6)], x.domain)
    assert _fro(A - LU) < res_tol                                   # A == L*U
    Lp = np.asarray(L(jnp.asarray(p)))                              # L(p, :)
    assert np.linalg.norm(np.diag(Lp) - 1.0) < 10 * tol
    assert np.linalg.norm(np.tril(Lp) - Lp, 2) < tril_tol   # MATLAB norm = 2-norm


class TestChebfunLu:
    def test_all_matlab_assertions(self):
        tol = ChebfunPref().chebfuneps
        _check(chebfun(lambda t: t), tol, tol, 10 * tol)            # pass(1)-(4)
        _check(chebfun(lambda t: t, domain=(-2.0, 3.0)), tol, 1e3 * tol,
               20 * tol)                                            # pass(5)-(8)
