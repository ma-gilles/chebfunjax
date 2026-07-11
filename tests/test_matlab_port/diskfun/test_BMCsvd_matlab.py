"""Port of MATLAB Chebfun tests/diskfun/test_BMCsvd.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_BMCsvd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="internal BMC svd")


class TestDiskfunBmcsvd:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
