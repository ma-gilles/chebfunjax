"""Port of MATLAB Chebfun tests/chebfun/test_dct.m (Fable 5).

FIXED: chebfun.dct/idct (utils.fasttransforms) added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_dct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np


class TestChebfunDct:
    def test_roundtrips_all_kinds(self):
        from chebfunjax.utils.fasttransforms import dct, idct
        r = np.random.default_rng(0).standard_normal(17)
        for k in (1, 2, 3, 4):
            np.testing.assert_allclose(idct(dct(r, k), k), r,
                                       atol=1e-12)
