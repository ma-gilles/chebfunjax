"""Port of MATLAB Chebfun tests/singfun/test_zeroSingFun.m (Opus 4.8).

MATLAB ``singfun.zeroSingFun()`` builds the canonical zero singfun (zero
smooth part, no exponents).

Provenance
----------
MATLAB source : tests/singfun/test_zeroSingFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.fun.singfun import Singfun


class TestSingfunZeroSingFun:
    def test_no_exponents(self):
        f = Singfun.zeroSingFun()
        assert not any(f.exponents)

    def test_trivial_smooth_part(self):
        f = Singfun.zeroSingFun()
        assert f.iszero()
