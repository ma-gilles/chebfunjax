"""Port of MATLAB Chebfun tests/chebfun2/test_padua.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_padua.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no Padua-points constructor")


class TestChebfun2Padua:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
