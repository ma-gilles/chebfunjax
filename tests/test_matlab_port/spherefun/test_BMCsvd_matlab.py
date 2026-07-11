"""Port of MATLAB Chebfun tests/spherefun/test_BMCsvd.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_BMCsvd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="BMC-structured svd internal; no public accessor")


class TestSpherefunBmcsvd:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
