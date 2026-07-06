"""Port of MATLAB Chebfun tests/deltafun/test_iszero.m (Opus 4.8).

chebfunjax's Deltafun has no ``iszero`` method (and no empty Deltafun), so every
assertion in this MATLAB test is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no iszero() method (and no empty Deltafun)"
)


class TestDeltafunIszero:
    def test_iszero_empty_and_zero_funpart(self):
        # pass(1): iszero(deltafun()) && iszero(deltafun(fun.constructor(0), []))
        pass

    def test_not_iszero_with_deltas(self):
        # pass(2): ~iszero(d) when d has non-zero delta magnitudes
        pass
