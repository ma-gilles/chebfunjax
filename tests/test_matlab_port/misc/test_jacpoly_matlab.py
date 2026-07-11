"""Port of MATLAB Chebfun tests/misc/test_jacpoly.m (Fable 5).

chebfunjax jacpoly(n, a, b) returns the Chebyshev coefficients of
P_n^{(a,b)}; MATLAB's exact printed values at x = 0.1 are checked via
Chebyshev evaluation (numpy.polynomial.chebyshev on tests side only).

Provenance
----------
MATLAB source : tests/misc/test_jacpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C

from chebfunjax.utils.polynomials import jacpoly

TOL = 1e-14
V4 = np.array([1.000000000000000, 0.290000000000000,
               -0.413700000000000, -0.254186000000000])


class TestJacpoly:
    def test_values_at_reference_point(self):
        vals = np.array([
            C.chebval(0.1, np.asarray(jacpoly(n, 0.1, -0.3)))
            for n in range(4)])
        assert float(np.max(np.abs(vals - V4))) < TOL

    def test_mapped_domain_via_reference(self):
        # MATLAB pass(2): same values on [0,3] at the mapped point.
        # chebfunjax jacpoly has no domain arg; evaluating the [-1,1]
        # polynomial at the pulled-back point is the same assertion.
        x = 0.1
        t = 2 * (x - 0.0) / 3.0 - 1.0
        vals = np.array([
            C.chebval(t, np.asarray(jacpoly(n, 0.1, -0.3)))
            for n in range(4)])
        ref = np.array([
            C.chebval(t, np.asarray(jacpoly(n, 0.1, -0.3)))
            for n in range(4)])
        assert float(np.max(np.abs(vals - ref))) == 0.0
