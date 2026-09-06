"""Port of MATLAB Chebfun tests/chebfun3/test_chebcoeffs3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_chebcoeffs3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebpoly
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, outer_prod
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _exact(m, n, p):
    # MATLAB grows zeros(m+1, n+1, p+1) on the out-of-range assignments.
    q = max(m, n, p) + 1
    C = np.zeros((q, q, q))
    C[m, m, m] = 1
    C[n, n, n] = 1
    C[p, p, p] = 1
    return C


class TestChebfun3Chebcoeffs3:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb3Prefs.chebfun3eps
        m, n, p = 8, 10, 5
        Tm, Tn, Tp = chebpoly(m), chebpoly(n), chebpoly(p)
        f = outer_prod(Tm, Tm, Tm) + outer_prod(Tn, Tn, Tn) + outer_prod(Tp, Tp, Tp)
        E = _exact(m, n, p)

        def _pad(C):
            out = np.zeros(np.maximum(C.shape, E.shape))
            out[:C.shape[0], :C.shape[1], :C.shape[2]] = C
            return out

        def _err(C):
            C = _pad(np.asarray(C))
            return np.linalg.norm((C - _pad(E)).ravel())

        assert _err(f.chebcoeffs3()) < tol                                  # pass(1)
        assert _err(f.coeffs3()) < tol                                      # pass(2)
        assert _err(Chebfun3.vals2coeffs(f.chebpolyval3())) < tol           # pass(3)
        core, cc, rc, tc = f.chebcoeffs3(low_rank=True)
        X = Chebfun3.txm(Chebfun3.txm(Chebfun3.txm(core, cc, 1), rc, 2), tc, 3)
        assert _err(X) < tol                                                # pass(4)
        core, cc, rc, tc = f.coeffs3(low_rank=True)
        X = Chebfun3.txm(Chebfun3.txm(Chebfun3.txm(core, cc, 1), rc, 2), tc, 3)
        assert _err(X) < tol                                                # pass(5)
