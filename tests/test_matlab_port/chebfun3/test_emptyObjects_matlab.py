"""Port of MATLAB Chebfun tests/chebfun3/test_emptyObjects.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no empty Chebfun3 representation")


class TestChebfun3Emptyobjects:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
