"""Port of MATLAB Chebfun tests/misc/test_cumsummat.m (Fable 5).

MATLAB compares against colloc classes; chebfunjax has one cumsummat.
The ported identities: interval scaling, and exactness of the
antiderivative on polynomial data (the property the colloc comparison
certifies).

Provenance
----------
MATLAB source : tests/misc/test_cumsummat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.diffmat import cumsummat
from chebfunjax.utils.quadrature import chebpts

TOL = 1e-10


class TestCumsummat:
    def test_interval_scaling(self):
        A = np.asarray(cumsummat(5))
        B = np.asarray(cumsummat(5, domain=(-2.0, 2.0)))
        assert float(np.max(np.abs(B - 2 * A))) < TOL

    def test_antiderivative_of_polynomial(self):
        n = 9
        x = np.asarray(chebpts(n, kind=2))
        Q = np.asarray(cumsummat(n))
        # antiderivative of 3x^2 vanishing at -1 is x^3 + 1
        got = Q @ (3 * x ** 2)
        assert float(np.max(np.abs(got - (x ** 3 + 1)))) < TOL
