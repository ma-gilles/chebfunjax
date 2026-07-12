"""Port of MATLAB Chebfun tests/chebfun/test_dlt.m (Fable 5).

FIXED: chebfun.dlt added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_dlt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np


class TestChebfunDlt:
    def test_legendre_series_evaluation(self):
        from scipy.special import eval_legendre

        from chebfunjax.utils.fasttransforms import dlt
        from chebfunjax.utils.quadrature import legpts
        c = np.zeros(9)
        c[4] = 1.0
        v = dlt(c)
        x, _ = (np.asarray(t) for t in legpts(9))
        np.testing.assert_allclose(v, eval_legendre(4, x), atol=1e-13)
