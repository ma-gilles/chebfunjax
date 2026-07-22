"""Port of MATLAB Chebfun tests/chebop2/test_subsref.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires N(m,n) returning the discretization matrix and N*f / N(f) applying the PDO to a chebfun2; only the N.coeffs accessor exists.")


class TestChebop2Subsref:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
