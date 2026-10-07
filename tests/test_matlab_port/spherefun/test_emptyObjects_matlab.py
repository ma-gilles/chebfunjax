"""Port of MATLAB Chebfun tests/spherefun/test_emptyObjects.m (Fable 5).

FIXED: empty Spherefun with empty propagation through the command set
added in the Fable 5 audit. The MATLAB test uses try/catch to require
no error; it does not require every result to be a Spherefun. This selection
still omits some of the original command list.

Provenance
----------
MATLAB source : tests/spherefun/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunEmptyobjects:
    def test_all_commands_tolerate_empty(self):
        f = Spherefun.empty()
        assert f.norm().shape == (0,)
        results = [f + f, f * 2, f ** 2, -f, f.laplacian(), f.cos()]
        # @spherefun/sum2.m returns scalar zero when idxPlus is empty.
        assert float(f.sum2()) == 0.0
        for r in results:
            assert hasattr(r, "isempty") and r.isempty()
