"""Port of MATLAB Chebfun tests/chebop/test_jumps_manual.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_jumps_manual.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="interior jump conditions not implemented")


class TestChebopJumpsManual:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
