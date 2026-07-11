"""Port of MATLAB Chebfun tests/diskfunv/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="diskfunv: 'feval' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestDiskfunvFeval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
