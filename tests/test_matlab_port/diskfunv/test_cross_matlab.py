"""Port of MATLAB Chebfun tests/diskfunv/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="diskfunv: 'cross' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestDiskfunvCross:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
