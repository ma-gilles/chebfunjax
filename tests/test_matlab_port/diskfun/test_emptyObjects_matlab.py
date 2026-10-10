"""Port of MATLAB Chebfun tests/diskfun/test_emptyObjects.m (Fable 5).

FIXED: empty Diskfun with empty propagation through the command set
added in the Fable 5 audit (the MATLAB test asserts every listed
command tolerates the empty object).

Provenance
----------
MATLAB source : tests/diskfun/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun


class TestDiskfunEmptyobjects:
    def test_all_commands_tolerate_empty(self):
        f = Diskfun.empty()
        results = [f + f, f * 2, f ** 2, -f, f.laplacian(), f.cos()]
        for r in results:
            assert r.isempty()
        # Native test_emptyObjects.m checks commands tolerate empty input,
        # not that numeric outputs are diskfuns. @diskfun/norm.m returns [].
        assert np.asarray(f.norm()).size == 0
        f.sum2()  # Native integral2/sum2 command must complete.
