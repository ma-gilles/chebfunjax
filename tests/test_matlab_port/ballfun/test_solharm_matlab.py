"""Port of MATLAB Chebfun tests/ballfun/test_solharm.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_solharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Ballfun has no solharm (solid harmonics constructor)")


class TestBallfunSolharm:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
