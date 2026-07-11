"""Port of MATLAB Chebfun tests/chebop/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop has no D*f operator application (linearize/apply not exposed)")


class TestChebopDiff:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
