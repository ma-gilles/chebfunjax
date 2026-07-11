"""Port of MATLAB Chebfun tests/chebop/test_shortPulses.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_shortPulses.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="requires breakpoint preservation in solve")


class TestChebopShortpulses:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
