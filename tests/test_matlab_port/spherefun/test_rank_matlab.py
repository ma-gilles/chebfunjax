"""Port of MATLAB Chebfun tests/spherefun/test_rank.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_rank.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun rank accessor exists but MATLAB's test needs plus/simplify (absent)")


class TestSpherefunRank:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
