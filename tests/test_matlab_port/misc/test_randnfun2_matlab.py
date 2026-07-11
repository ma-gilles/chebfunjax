"""Port of MATLAB Chebfun tests/misc/test_randnfun2.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfun2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no randnfun2 (random chebfun2)")


class TestMiscRandnfun2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
