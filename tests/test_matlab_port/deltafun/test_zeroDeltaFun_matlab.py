"""Port of MATLAB Chebfun tests/deltafun/test_zeroDeltaFun.m (Opus 4.8).

MATLAB provides a static ``deltafun.zeroDeltaFun([a, b])`` factory that builds a
Deltafun with a zero funPart on [a, b] and no deltas.  chebfunjax has no such
factory method, so every assertion is skipped with a precise reason.  (The
underlying object is constructible via
``Deltafun.from_fun(Bndfun.from_function(lambda x: 0*x, Domain((a, b))))``, but
the named factory being tested does not exist.)

Provenance
----------
MATLAB source : tests/deltafun/test_zeroDeltaFun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no zeroDeltaFun() factory method"
)


class TestDeltafunZeroDeltaFun:
    def test_default_funpart_zero(self):
        # pass(1): iszero(d.funPart)
        pass

    def test_default_domain(self):
        # pass(2): d.funPart.domain == [-1, 1]
        pass

    def test_default_no_delta_mag(self):
        # pass(3): isempty(d.deltaMag)
        pass

    def test_default_no_delta_loc(self):
        # pass(4): isempty(d.deltaLoc)
        pass

    def test_custom_funpart_zero(self):
        # pass(5): iszero(d.funPart) for zeroDeltaFun([4,5])
        pass

    def test_custom_domain(self):
        # pass(6): d.funPart.domain == [4, 5]
        pass

    def test_custom_no_delta_mag(self):
        # pass(7): isempty(d.deltaMag)
        pass

    def test_custom_no_delta_loc(self):
        # pass(8): isempty(d.deltaLoc)
        pass
