"""Port of MATLAB Chebfun tests/spherefun/test_coeffs2vals_vals2coeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_coeffs2vals_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun 2D coefficient transforms not exposed")


class TestSpherefunCoeffs2valsVals2coeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
