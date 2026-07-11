"""Port of MATLAB Chebfun tests/chebfun/test_deltaOps.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_deltaOps.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="delta-function chebfun ops (dirac arithmetic at the chebfun level) limited to diff/sum; deltafun layer tested separately")


class TestChebfunDeltaops:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
