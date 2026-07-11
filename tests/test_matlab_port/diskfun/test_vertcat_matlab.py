"""Port of MATLAB Chebfun tests/diskfun/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no diskfunv vertcat")


class TestDiskfunVertcat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
