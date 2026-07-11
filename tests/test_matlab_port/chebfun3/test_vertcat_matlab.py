"""Port of MATLAB Chebfun tests/chebfun3/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun3v vertical concatenation")


class TestChebfun3Vertcat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
