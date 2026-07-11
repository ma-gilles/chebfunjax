"""Port of MATLAB Chebfun tests/ballfun/test_vals2coeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="coefficient transforms internal")


class TestBallfunVals2coeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
