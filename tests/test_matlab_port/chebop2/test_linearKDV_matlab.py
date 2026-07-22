"""Port of MATLAB Chebfun tests/chebop2/test_linearKDV.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_linearKDV.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires a 3rd-order-in-x term diffx(u,3) together with multi-condition BC rbc=@(t,u)[u-...;diff(u)-...]; unavailable in the value-space solver.")


class TestChebop2Linearkdv:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
