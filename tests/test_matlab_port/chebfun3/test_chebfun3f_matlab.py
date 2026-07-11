"""Port of MATLAB Chebfun tests/chebfun3/test_chebfun3f.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_chebfun3f.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebfun3f (alternative constructor) variant")


class TestChebfun3Chebfun3f:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
