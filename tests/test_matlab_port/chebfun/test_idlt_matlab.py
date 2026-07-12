"""Port of MATLAB Chebfun tests/chebfun/test_idlt.m (Fable 5).

FIXED: chebfun.idlt added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_idlt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np


class TestChebfunIdlt:
    def test_inverse_roundtrip(self):
        from chebfunjax.utils.fasttransforms import dlt, idlt
        r = np.random.default_rng(2).standard_normal(21)
        np.testing.assert_allclose(idlt(dlt(r)), r, atol=1e-12)
