"""Port of MATLAB Chebfun tests/chebfun2/test_emptyObjects.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no empty Chebfun2 representation")


class TestChebfun2Emptyobjects:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
