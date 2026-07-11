"""Port of MATLAB Chebfun tests/misc/test_padeapprox.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_padeapprox.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.ratapprox import padeapprox

TOL = 1e-10


class TestPadeapprox:
    def test_rational_with_known_poles(self):
        r, a, b, mu, nu, poles, residues = padeapprox(
            lambda x: (x ** 4 - 3) / ((x + 3.2) * (x - 2.2)), 10, 10)
        assert mu == 4 and nu == 2
        p = np.sort_complex(np.asarray(poles))
        assert float(np.max(np.abs(p - np.array([-3.2, 2.2])))) < TOL

    def test_coefficient_input(self):
        r, a, b, *_ = padeapprox(np.array([1.0, 1.0j]), 0, 1)
        assert abs(np.asarray(a).ravel()[0] - 1) < 1e-10
        assert abs(np.asarray(b).ravel()[1] - (-1j)) < TOL

    def test_degenerate_with_tolerance(self):
        r, a, b, mu, nu, poles, residues = padeapprox(
            lambda x: x / (1 - x), 5, 6, r=0.5)
        assert mu == 1 and nu == 1
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        assert float(np.max(np.abs(a - np.array([0.0, 1.0])))) < TOL
        assert float(np.max(np.abs(b - np.array([1.0, -1.0])))) < TOL
        assert abs(complex(np.asarray(poles).ravel()[0]) - 1) < TOL
        assert abs(complex(np.asarray(residues).ravel()[0]) + 1) < TOL
