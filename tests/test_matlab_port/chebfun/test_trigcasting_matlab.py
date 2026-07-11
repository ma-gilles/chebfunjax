"""Port of MATLAB Chebfun tests/chebfun/test_trigcasting.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_trigcasting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="trig/cheb mixed-arithmetic casting rules not implemented")


class TestChebfunTrigcasting:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
