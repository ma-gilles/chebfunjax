"""Port of MATLAB Chebfun tests/chebfun2/test_emptyObjects.m
(Fable 5).

FIXED: empty Chebfun2 (Chebfun2.empty()) with empty propagation
through the command set added in the Fable 5 audit -- every listed
operation on the empty object returns the empty object instead of
crashing, which is exactly what the MATLAB try-block asserts.

Provenance
----------
MATLAB source : tests/chebfun2/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Emptyobjects:
    def test_all_commands_tolerate_empty(self):
        f = Chebfun2.empty()
        results = [
            f + f, f * 2, f ** 2, f.sqrt(), f.sum(), f.norm(),
            f.squeeze(), f.diff(), f.cos(), f.sin(),
            (f ** 2) + f, f.diag_fun(), f.trace(), f.mean(),
            f.mean2(), f.minandmax2(), f.fliplr(), f.flipud(),
            f.cumsum(), -f,
        ]
        for r in results:
            assert hasattr(r, "isempty") and r.isempty()
