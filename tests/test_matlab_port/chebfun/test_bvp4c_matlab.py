"""Port of MATLAB Chebfun tests/chebfun/test_bvp4c.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_bvp4c.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no bvp4c wrapper (chebop.solve covers BVPs)")


class TestChebfunBvp4c:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
