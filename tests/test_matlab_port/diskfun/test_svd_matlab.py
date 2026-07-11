"""Port of MATLAB Chebfun tests/diskfun/test_svd.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_svd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no svd")


class TestDiskfunSvd:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
