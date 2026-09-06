"""Port of MATLAB Chebfun tests/chebfun2/test_chol.m (Fable 5).

``R = chol(f)`` is a Quasimatrix of the rows ``R_j(x)``; ``R' * R`` is
``Chebfun2.from_cdr(R, 1, R)``; ``R(:, p)`` is the matrix of the rows
evaluated at the points ``p``.

Provenance
----------
MATLAB source : tests/chebfun2/test_chol.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.quadrature import chebpts_ab

jax.config.update("jax_enable_x64", True)


def _check(f, x, tol):
    R = f.chol()
    g = Chebfun2.from_cdr(R.cols, np.ones(R.n_cols), R.cols, f.domain)   # R' * R
    piv = np.asarray(f.pivot_locations, dtype=float)
    assert np.linalg.norm(piv[:, 0] - piv[:, 1]) < tol
    assert np.linalg.norm(np.asarray(f.fevalm(x, x)) - np.asarray(g.fevalm(x, x))) < tol
    r1 = float(R.cols[0](jnp.asarray(piv[0, 1])))
    assert abs(r1 - np.sqrt(float(f(piv[0, 0], piv[0, 1])))) < tol
    p = jnp.asarray(piv[:, 0])
    Rp = np.asarray(R(p))                       # R(:, p) evaluated: (npts, k)
    assert np.linalg.norm(np.diag(Rp @ Rp.T) - np.asarray(f(p, p))) < tol


class TestChebfun2Chol:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        x = jnp.asarray(chebpts_ab(100, -1.0, 1.0))
        f = chebfun2(lambda x, y: 1.0 / (10 + x ** 2 + y ** 2))
        _check(f, x, tol)                                            # pass(1)-(4)

        x = jnp.asarray(chebpts_ab(100, -2.1, 4.3))
        f = chebfun2(lambda x, y: 1.0 / (10 + x ** 2 + y ** 2),
                     domain=(-2.1, 4.3, -2.1, 4.3))
        _check(f, x, tol)                                            # pass(5)-(8)

        xc = chebfun(lambda t: t)
        A = Quasimatrix([xc ** 2, xc ** 4], xc.domain)
        B = Chebfun2.from_cdr(A.cols, np.ones(2), A.cols, (-1.0, 1.0, -1.0, 1.0))  # A * A'
        R = B.chol()
        RR = Chebfun2.from_cdr(R.cols, np.ones(R.n_cols), R.cols, B.domain)
        xx = jnp.linspace(-1, 1, 40)
        err = float(np.max(np.abs(np.asarray(RR.fevalm(xx, xx)) - np.asarray(B.fevalm(xx, xx)))))
        assert err < tol                                             # pass(9)
