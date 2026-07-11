"""Port of MATLAB Chebfun tests/chebop/test_jump_scaled.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_jump_scaled.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="interior jump conditions not implemented")


class TestChebopJumpScaled:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
