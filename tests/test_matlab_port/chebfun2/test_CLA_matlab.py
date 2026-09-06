"""Port of MATLAB Chebfun tests/chebfun2/test_CLA.m (Fable 5).

``F.cols``/``F.rows`` are Quasimatrices of the CDR slices; ``qr`` is the
continuous Householder QR of a quasimatrix; the reconstruction
``(C*D)*R.'`` is ``Chebfun2.from_cdr``.

Provenance
----------
MATLAB source : tests/chebfun2/test_CLA.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _qr_residual(Q):
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    q, R = Q.qr()
    cols = list(getattr(q, "cols", q))
    QR = Quasimatrix(cols, Q.domain) @ np.asarray(R)
    return float(np.sqrt(sum(float((a - b).norm(2)) ** 2
                             for a, b in zip(QR.cols, Q.cols))))


class TestChebfun2CLA:
    def test_all_matlab_assertions(self):
        tol = 200 * ChebfunPref().cheb2Prefs.chebfun2eps
        rng = np.random.RandomState(0)
        gam = 10
        centres = [(2 * rng.rand() - 1, 2 * rng.rand() - 1) for _ in range(20)]

        def f(x, y):
            out = 0.0 * x
            for x0, y0 in centres:
                out = out + jnp.exp(-gam * ((x - x0) ** 2 + (y - y0) ** 2))
            return out
        F = chebfun2(f)
        assert _qr_residual(F.cols) < tol                           # pass(1)
        assert _qr_residual(F.rows) < tol                           # pass(2)

        C, D, R = F.cdr()
        D = np.array(D)
        C = list(C)
        for k in range(D.shape[0]):
            if D[k, k] < 0:
                D[k, k] = -D[k, k]
                C[k] = -C[k]
        G1 = Chebfun2.from_cdr(C, D, R, F.domain)                  # (C*D)*R.'
        assert float((G1 - F).norm()) < tol                          # pass(3)
        assert float((F - G1).norm()) < tol                          # pass(4)
        G2 = Chebfun2.from_cdr(C, D, R, F.domain)                  # C*(D*R.')
        assert float((G2 - F).norm()) < tol                          # pass(5)
        assert float((F - G2).norm()) < tol                          # pass(6)
        sq = np.sqrt(np.diag(D))
        G3 = Chebfun2.from_cdr([c * float(s) for c, s in zip(C, sq)],
                               np.ones(len(C)),
                               [r * float(s) for r, s in zip(R, sq)],
                               F.domain)                             # (C*sqrt(D))*(sqrt(D)*R.')
        assert float((G3 - F).norm()) < tol                          # pass(7)
        assert float((F - G3).norm()) < tol                          # pass(8)
