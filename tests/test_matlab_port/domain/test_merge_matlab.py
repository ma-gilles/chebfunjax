"""Port of MATLAB Chebfun tests/domain/test_merge.m (Fable 5).

FIXED: domain.merge added in the Fable 5 audit (sorted union of
breakpoint vectors, infinite endpoints preserved).

Provenance
----------
MATLAB source : tests/domain/test_merge.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.domain import merge


class TestDomainMerge:
    def test_breakpoint_union(self):
        d = merge([-np.inf, 1, 10, np.inf],
                  [-np.inf, 1, 5, np.inf])
        assert d == [-np.inf, 1.0, 5.0, 10.0, np.inf]
