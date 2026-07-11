"""Port of MATLAB Chebfun tests/diskfun/test_diag.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no diag")


class TestDiskfunDiag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
