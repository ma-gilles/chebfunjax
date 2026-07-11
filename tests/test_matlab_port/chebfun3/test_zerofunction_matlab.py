"""Port of MATLAB Chebfun tests/chebfun3/test_zerofunction.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_zerofunction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax cannot represent the zero-rank Chebfun3 the file tests")


class TestChebfun3Zerofunction:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
