"""Port of MATLAB Chebfun tests/chebfun2/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun2v vertical concatenation of chebfun2s")


class TestChebfun2Vertcat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
