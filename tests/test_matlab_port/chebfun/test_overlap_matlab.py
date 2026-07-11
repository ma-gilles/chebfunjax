"""Port of MATLAB Chebfun tests/chebfun/test_overlap.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_overlap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no overlap (domain merging is internal)")


class TestChebfunOverlap:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
