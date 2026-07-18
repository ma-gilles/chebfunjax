"""Port of MATLAB Chebfun tests/diskfun/test_roots.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="diskfun roots requires 2D zero-contour "
                              "extraction as parametrized complex-valued "
                              "chebfuns (chebfun2 roots@separableApprox: "
                              "marching-squares tracing + curve fitting), a "
                              "subsystem not present in chebfunjax -- out of "
                              "scope for the flip/abs/rotate/sum/optimization "
                              "gap batch (Fable 5)")


class TestDiskfunRoots:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
