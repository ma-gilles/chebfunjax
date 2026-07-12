"""Port of MATLAB Chebfun tests/chebfun/test_prod.m (Fable 5).

FIXED: Chebfun.prod added in the Fable 5 audit.  Array-valued cases
remain skipped (no multi-column chebfun).

Provenance
----------
MATLAB source : tests/chebfun/test_prod.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

import chebfunjax as cj

TOL = 100 * np.finfo(float).eps


class TestChebfunProd:
    def test_scalar_valued(self):
        # pass(1): prod(x+2) = exp(int log(x+2)) = 27 e^-2
        f = cj.chebfun(lambda x: x + 2)
        assert abs(float(f.prod()) - 27 * np.exp(-2)) < TOL
