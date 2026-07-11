"""Port of MATLAB Chebfun tests/misc/test_ultrapoly.m (Fable 5).

roots(C_n^{(lam)}) must equal the Gauss-Jacobi nodes
jacpts(n, lam-1/2, lam-1/2).

Provenance
----------
MATLAB source : tests/misc/test_ultrapoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C

from chebfunjax.utils.polynomials import ultrapoly
from chebfunjax.utils.quadrature import jacpts

TOL = 1e-12


class TestUltrapoly:
    def test_roots_match_jacpts_even(self):
        lam = 2.1
        r = np.sort(np.real(C.chebroots(np.asarray(ultrapoly(10, lam)))))
        exact = np.sort(np.asarray(jacpts(10, lam - 0.5, lam - 0.5)[0]))
        assert float(np.max(np.abs(r - exact))) < TOL

    def test_roots_match_jacpts_odd(self):
        lam = 1.9
        r = np.sort(np.real(C.chebroots(np.asarray(ultrapoly(11, lam)))))
        exact = np.sort(np.asarray(jacpts(11, lam - 0.5, lam - 0.5)[0]))
        assert float(np.max(np.abs(r - exact))) < TOL
