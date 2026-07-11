"""Port of MATLAB Chebfun tests/diskfun/test_projection.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_projection.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="BMC projection internal")


class TestDiskfunProjection:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
