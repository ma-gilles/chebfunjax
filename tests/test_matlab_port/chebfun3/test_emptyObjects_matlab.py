"""Port of MATLAB Chebfun tests/chebfun3/test_emptyObjects.m (Fable 5).

FIXED: empty Chebfun3 with empty propagation through the command set
added in the Fable 5 audit (the MATLAB test asserts every listed
command tolerates the empty object).

Provenance
----------
MATLAB source : tests/chebfun3/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Emptyobjects:
    def test_all_commands_tolerate_empty(self):
        f = Chebfun3.empty()
        results = [f + f, f * 2, f ** 2, -f, f.sum3(), f.norm(), f.squeeze(), f.cos(), f.mean3()]
        for r in results:
            assert hasattr(r, "isempty") and r.isempty()
