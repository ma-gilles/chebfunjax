"""Port of MATLAB Chebfun tests/chebfun2/test_zerofunction.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_zerofunction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax cannot represent the empty/zero-rank Chebfun2 the file tests")


class TestChebfun2Zerofunction:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
