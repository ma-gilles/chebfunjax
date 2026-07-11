"""Port of MATLAB Chebfun tests/chebfun/test_ultracoeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_ultracoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebfun has no ultracoeffs (ultra2ultra tested in misc)")


class TestChebfunUltracoeffs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
