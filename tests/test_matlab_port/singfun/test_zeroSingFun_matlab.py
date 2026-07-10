"""Port of MATLAB Chebfun tests/singfun/test_zeroSingFun.m (Opus 4.8).

MATLAB ``singfun.zeroSingFun()`` builds the canonical zero singfun (zero
smooth part, no exponents).  chebfunjax has no ``zeroSingFun`` factory nor an
``iszero`` predicate, so both assertions are skipped.

Provenance
----------
MATLAB source : tests/singfun/test_zeroSingFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestSingfunZeroSingFun:
    def test_no_exponents(self):
        pytest.skip("chebfunjax has no singfun.zeroSingFun() factory")

    def test_trivial_smooth_part(self):
        pytest.skip("chebfunjax has no singfun.zeroSingFun() factory / iszero")
