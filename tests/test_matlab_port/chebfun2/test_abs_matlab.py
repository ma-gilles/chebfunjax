"""Port of MATLAB Chebfun tests/chebfun2/test_abs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no abs()")


class TestChebfun2Abs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
