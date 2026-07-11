"""Port of MATLAB Chebfun tests/chebfun/test_jaccoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_jaccoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no jaccoeffs (cheb2jac transform tested in misc)")


class TestChebfunJaccoeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
