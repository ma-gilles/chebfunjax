"""Port of MATLAB Chebfun tests/chebfun2/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no integral(f, curve) line-integral method")


class TestChebfun2Integral:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
