"""Port of MATLAB Chebfun tests/misc/test_minimax.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_minimax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test operates on chebfun inputs incl. cf() comparison; chebfunjax minimax(callable) is covered by unit tests (NOT YET PORTED assertion-for-assertion)")


class TestMiscMinimax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
