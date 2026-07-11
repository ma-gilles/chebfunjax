"""Port of MATLAB Chebfun tests/chebfun/test_legcoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_legcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no legcoeffs (cheb2leg transform tested in misc)")


class TestChebfunLegcoeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
