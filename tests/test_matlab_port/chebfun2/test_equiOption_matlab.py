"""Port of MATLAB Chebfun tests/chebfun2/test_equiOption.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_equiOption.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 constructor has no 'equi' (equispaced) option")


class TestChebfun2Equioption:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
