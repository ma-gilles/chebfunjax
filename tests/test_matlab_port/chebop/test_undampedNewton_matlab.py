"""Port of MATLAB Chebfun tests/chebop/test_undampedNewton.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_undampedNewton.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="damping-off pref not exposed")


class TestChebopUndampednewton:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
