"""Port of MATLAB Chebfun tests/chebfun2/test_std.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_std.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="test uses the dimensional std(f, [], dim) which returns a Chebfun of the per-row/column standard deviation; chebfunjax has the scalar std2() but not the dimensional std() -- src gap")


class TestChebfun2Std:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
