"""Port of MATLAB Chebfun tests/ballfun/test_coeffs3.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_coeffs3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no coeffs3 accessor")


class TestBallfunCoeffs3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
