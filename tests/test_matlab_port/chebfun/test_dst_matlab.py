"""Port of MATLAB Chebfun tests/chebfun/test_dst.m (Fable 5).

FIXED: chebfun.dst/idst added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_dst.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np


class TestChebfunDst:
    def test_roundtrips_all_kinds(self):
        from chebfunjax.utils.fasttransforms import dst, idst
        r = np.random.default_rng(1).standard_normal(17)
        for k in (1, 2, 3, 4):
            np.testing.assert_allclose(idst(dst(r, k), k), r,
                                       atol=1e-12)
