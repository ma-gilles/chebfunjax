"""Port of MATLAB Chebfun tests/chebop/test_paramODE_inBCs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_inBCs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax chebop is scalar-only (no systems of ODEs / chebmatrix operators)")


class TestChebopParamodeInbcs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
