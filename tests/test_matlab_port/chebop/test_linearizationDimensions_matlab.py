"""Port of MATLAB Chebfun tests/chebop/test_linearizationDimensions.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_linearizationDimensions.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="this test inspects the block TYPES of a linearized system (linop(L).blocks, checking each is an operatorBlock vs a chebfun); chebfunjax builds system collocation matrices directly and exposes no linop/block-type introspection -- src gap (no counterpart)")


class TestChebopLinearizationdimensions:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
