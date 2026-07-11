"""Port of MATLAB Chebfun tests/ballfun/test_isequal.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no isequal")


class TestBallfunIsequal:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
