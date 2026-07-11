"""Port of MATLAB Chebfun tests/chebfun3/test_equiFlag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_equiFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 constructor has no 'equi' option")


class TestChebfun3Equiflag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
