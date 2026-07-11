"""Port of MATLAB Chebfun tests/chebfun2/test_minandmax2est.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no minandmax2est")


class TestChebfun2Minandmax2est:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
