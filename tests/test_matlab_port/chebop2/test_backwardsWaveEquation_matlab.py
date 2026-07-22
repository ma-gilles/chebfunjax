"""Port of MATLAB Chebfun tests/chebop2/test_backwardsWaveEquation.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_backwardsWaveEquation.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires a two-condition BC on one edge ubc=@(x,u)[u-...;diff(u)-...] (value + normal derivative) for the 2nd-order-in-time wave IVP; the value-space solver imposes a single Dirichlet condition per edge.")


class TestChebop2Backwardswaveequation:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
