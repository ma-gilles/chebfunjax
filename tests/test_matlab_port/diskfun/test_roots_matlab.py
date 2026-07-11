"""Port of MATLAB Chebfun tests/diskfun/test_roots.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no roots (zero contours)")


class TestDiskfunRoots:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
