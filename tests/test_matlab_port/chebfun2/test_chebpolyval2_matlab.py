"""Port of MATLAB Chebfun tests/chebfun2/test_chebpolyval2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_chebpolyval2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no chebpolyval2 (values matrix) accessor")


class TestChebfun2Chebpolyval2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
