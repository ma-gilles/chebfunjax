"""Port of MATLAB Chebfun tests/cheb/test_normal2.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_normal2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no cheb.normal2")


class TestChebNormal2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
