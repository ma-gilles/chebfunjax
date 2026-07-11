"""Port of MATLAB Chebfun tests/misc/test_diffmat.m (Fable 5).

Interval scaling identities and exactness on polynomials; the colloc1/
rectangular variants are skipped (single discretization).

Provenance
----------
MATLAB source : tests/misc/test_diffmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts

TOL = 1e-10


class TestMiscDiffmat:
    def test_interval_scaling_first_order(self):
        A = np.asarray(diffmat(5))
        B = np.asarray(diffmat(5, domain=(-2.0, 2.0)))
        assert float(np.max(np.abs(B - A / 2))) < TOL

    def test_interval_scaling_second_order(self):
        A = np.asarray(diffmat(5, 2))
        B = np.asarray(diffmat(5, 2, domain=(-2.0, 2.0)))
        assert float(np.max(np.abs(B - A / 4))) < TOL

    def test_exact_on_polynomial(self):
        n = 9
        x = np.asarray(chebpts(n, kind=2))
        D = np.asarray(diffmat(n))
        got = D @ (x ** 4)
        assert float(np.max(np.abs(got - 4 * x ** 3))) < TOL

    def test_rectangular_variants(self):
        pytest.skip("rectangular/colloc1 diffmat variants not "
                    "implemented (single square colloc2 discretization)")
