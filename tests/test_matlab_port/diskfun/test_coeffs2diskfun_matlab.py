"""Port of MATLAB Chebfun tests/diskfun/test_coeffs2diskfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_coeffs2diskfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no coefficient constructor")


class TestDiskfunCoeffs2diskfun:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
