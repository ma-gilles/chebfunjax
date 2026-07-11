"""Port of MATLAB Chebfun tests/diskfun/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no line-integral method")


class TestDiskfunIntegral:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
