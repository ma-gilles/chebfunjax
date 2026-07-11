"""Port of MATLAB Chebfun tests/ballfun/test_diskfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_diskfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no Ballfun->Diskfun slice extraction")


class TestBallfunDiskfun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
