"""Port of MATLAB Chebfun tests/diskfun/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Diskfun has no arithmetic (same gap class as Spherefun)")


class TestDiskfunTimes:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
