"""Port of MATLAB Chebfun tests/diskfun/test_coeffs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_coeffs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no coeffs2 accessor")


class TestDiskfunCoeffs2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
