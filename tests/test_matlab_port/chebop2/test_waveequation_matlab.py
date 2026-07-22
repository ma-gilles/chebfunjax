"""Port of MATLAB Chebfun tests/chebop2/test_waveequation.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_waveequation.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires a two-condition initial BC dbc=@(x,u)[u-...;diff(u)-...] for the 2nd-order-in-time wave IVP; unavailable in the value-space Dirichlet solver.")


class TestChebop2Waveequation:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
