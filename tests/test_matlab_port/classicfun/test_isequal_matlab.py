"""Port of MATLAB Chebfun tests/classicfun/test_isequal.m (Fable 5).

Provenance
----------
MATLAB source : tests/classicfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Classicfun has no isequal method")


class TestClassicfunIsequal:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
