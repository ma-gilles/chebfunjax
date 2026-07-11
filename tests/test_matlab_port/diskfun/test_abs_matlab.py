"""Port of MATLAB Chebfun tests/diskfun/test_abs.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Diskfun has no abs")


class TestDiskfunAbs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
