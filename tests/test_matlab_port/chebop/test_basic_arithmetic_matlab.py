"""Port of MATLAB Chebfun tests/chebop/test_basic_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_basic_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop objects have no + / - arithmetic or direct application A(u)")


class TestChebopBasic_arithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
