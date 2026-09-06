"""Port of MATLAB Chebfun tests/misc/test_chebpolyvalm.m (Fable 5).

``chebtech2({[], p}); poly(f)`` (monomial coefficients of a Chebyshev
series, highest degree first) is ``numpy.polynomial.chebyshev.cheb2poly``
reversed; ``polyvalm`` is ``numpy.polynomial.polynomial.polyvalfromroots``'s
matrix cousin, implemented by Horner here.

Provenance
----------
MATLAB source : tests/misc/test_chebpolyvalm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from numpy.polynomial import chebyshev as _C

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.polynomials import chebpolyvalm

jax.config.update("jax_enable_x64", True)


def _polyvalm(p_high_first, A):
    out = np.zeros_like(A)
    for c in p_high_first:
        out = out @ A + c * np.eye(A.shape[0])
    return out


class TestMiscChebpolyvalm:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(0)
        p_cheb = rng.rand(3) + rng.rand(3) * 1j
        A = rng.rand(3, 3) + rng.rand(3, 3) * 1j
        p_poly = _C.cheb2poly(p_cheb)[::-1]               # poly(f): highest first
        tol = 100 * ChebfunPref().chebfuneps
        err = np.linalg.norm(_polyvalm(p_poly, A)
                             - np.asarray(chebpolyvalm(p_cheb[::-1], A)))
        assert err < tol                                            # pass(1)
        err = np.linalg.norm(_polyvalm(p_poly, A)
                             - np.asarray(chebpolyvalm(p_cheb[::-1].reshape(1, -1), A)))
        assert err < tol                                            # pass(2)

        with pytest.raises(ValueError, match="chebpolyvalm:square"):
            chebpolyvalm(p_cheb, rng.rand(3, 4))                    # pass(3)
        with pytest.raises(ValueError, match="chebpolyvalm:vector"):
            chebpolyvalm(rng.rand(2, 3), A)                         # pass(4)
