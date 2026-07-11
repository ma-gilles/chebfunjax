"""Port of MATLAB Chebfun tests/diskfun/test_get.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB get() interface")


class TestDiskfunGet:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
