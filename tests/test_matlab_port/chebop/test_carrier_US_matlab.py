"""Port of MATLAB Chebfun tests/chebop/test_carrier_US.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_carrier_US.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="discretization-variant file (colloc1/ultraS); chebfunjax has a single collocation discretization -- the _C2 case is the ported one")


class TestChebopCarrierUs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
