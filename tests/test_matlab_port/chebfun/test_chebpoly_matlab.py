"""Port of MATLAB Chebfun tests/chebfun/test_chebpoly.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_chebpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB chebpoly(f) quasimatrix accessor; coefficients covered by chebcoeffs port")


class TestChebfunChebpoly:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
