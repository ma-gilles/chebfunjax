"""Port of MATLAB Chebfun tests/diskfunv/test_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="diskfunv: 'syntax' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestDiskfunvSyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
