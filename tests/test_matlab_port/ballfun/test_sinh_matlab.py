"""Port of MATLAB Chebfun tests/ballfun/test_sinh.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sinh.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no sinh composition")


class TestBallfunSinh:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
