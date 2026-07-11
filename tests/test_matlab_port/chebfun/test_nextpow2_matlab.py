"""Port of MATLAB Chebfun tests/chebfun/test_nextpow2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_nextpow2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no nextpow2")


class TestChebfunNextpow2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
